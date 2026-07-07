#!/usr/bin/env python3
"""
Score benign/pathogenic missense variants with ESM2-650M.

Input:
  1) A VEP-style tab-separated file containing at least:
     #Uploaded_variation, Protein_position, Amino_acids, CLIN_SIG
     The script also uses MANE, MANE_SELECT, TSL, APPRIS when present to choose
     the best transcript row per variant.
  2) A FASTA file with the reference protein sequence matching the positions
     you want to score, typically the MANE/canonical protein sequence.

Output:
  <out_prefix>.scores.tsv          one row per selected variant
  <out_prefix>.summary.tsv         summary by benign/pathogenic class
  <out_prefix>.skipped.tsv         rows/variants skipped and why
  <out_prefix>.selection_warnings.tsv  duplicated IDs with multiple changes
  <out_prefix>.llr_hist.png        log-likelihood-ratio distribution
  <out_prefix>.llr_boxplot.png     boxplot of log-likelihood-ratio

Scoring:
  For each variant, the residue position is masked and ESM2 gives:
      P(mutant amino acid | masked sequence)
      P(wild-type amino acid | masked sequence)

  log_likelihood_ratio = log P(mutant) - log P(wild-type)
  fold_change          = P(mutant) / P(wild-type) = exp(log_likelihood_ratio)

  Negative log_likelihood_ratio means the mutant amino acid is less likely than
  the reference amino acid in that sequence context.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class ParsedVariant:
    variant_id: str
    clinical_class: str
    protein_position: int
    ref_aa: str
    alt_aa: str
    original_label: str


def read_fasta(path: str) -> str:
    """Read a one-sequence FASTA file and return an uppercase protein sequence."""
    seq_parts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_parts.append(line)
    seq = "".join(seq_parts).upper().replace(" ", "").replace("*", "")
    if not seq:
        raise ValueError(f"No sequence found in FASTA: {path}")
    non_standard = sorted(set(seq) - STANDARD_AA - {"X"})
    if non_standard:
        warnings.warn(
            f"Reference sequence contains non-standard symbols: {non_standard}. "
            "Variants at those positions will be skipped if needed."
        )
    return seq


def normalise_clinsig(label: str, strict: bool = False) -> Optional[str]:
    """
    Convert diverse ClinVar/VEP CLIN_SIG strings into 'benign', 'pathogenic', or None.

    Lenient default:
      - benign if there is any benign/likely_benign term and no pathogenic term
      - pathogenic if there is any pathogenic/likely_pathogenic term and no benign term
      - ignore uncertain/not_provided/conflicting terms unless both benign and pathogenic occur

    Strict mode:
      - skip labels containing uncertain/conflicting/not_provided/etc.
      - classify only clean benign-like or pathogenic-like labels
    """
    if label is None or pd.isna(label):
        return None

    s = str(label).strip().lower()
    if not s or s == "-":
        return None

    s = s.replace(" ", "_").replace("-", "_")
    parts = [p.strip() for p in re.split(r"[,;|]+", s) if p.strip()]

    ambiguous_keywords = (
        "uncertain",
        "vus",
        "conflicting",
        "not_provided",
        "not_specified",
        "association",
        "risk_factor",
        "drug_response",
        "protective",
        "affects",
        "other",
    )

    if strict and any(any(k in p for k in ambiguous_keywords) for p in parts):
        return None

    has_benign = any("benign" in p for p in parts)
    # Exclude the word pathogenicity in 'conflicting_classifications_of_pathogenicity'
    has_pathogenic = any(
        ("pathogenic" in p and "conflicting_classifications" not in p) for p in parts
    )

    if has_benign and not has_pathogenic:
        return "benign"
    if has_pathogenic and not has_benign:
        return "pathogenic"
    return None


def parse_single_missense_row(row: pd.Series, strict_labels: bool) -> Tuple[Optional[ParsedVariant], Optional[str]]:
    """Parse a VEP row into a single amino-acid substitution."""
    variant_id = str(row.get("#Uploaded_variation", "")).strip()
    label = str(row.get("CLIN_SIG", "")).strip()
    clinical_class = normalise_clinsig(label, strict=strict_labels)
    if clinical_class is None:
        return None, "ambiguous_or_unsupported_CLIN_SIG"

    pos_raw = str(row.get("Protein_position", "")).strip()
    aa_raw = str(row.get("Amino_acids", "")).strip().upper()

    if not variant_id:
        return None, "missing_variant_id"

    if not re.fullmatch(r"\d+", pos_raw):
        return None, "not_single_protein_position"
    protein_position = int(pos_raw)

    if "/" not in aa_raw:
        return None, "missing_amino_acid_change"
    ref_aa, alt_aa = aa_raw.split("/", 1)
    ref_aa = ref_aa.strip()
    alt_aa = alt_aa.strip()

    if len(ref_aa) != 1 or len(alt_aa) != 1:
        return None, "not_single_amino_acid_substitution"
    if ref_aa not in STANDARD_AA or alt_aa not in STANDARD_AA:
        return None, "non_standard_amino_acid"
    if ref_aa == alt_aa:
        return None, "same_ref_and_alt_amino_acid"

    return ParsedVariant(
        variant_id=variant_id,
        clinical_class=clinical_class,
        protein_position=protein_position,
        ref_aa=ref_aa,
        alt_aa=alt_aa,
        original_label=label,
    ), None


def tsl_rank(value: object) -> int:
    """Lower TSL values are better. Missing/non-numeric is ranked poorly."""
    try:
        return int(str(value).split()[0])
    except Exception:
        return 999


def appris_rank(value: object) -> int:
    """Basic APPRIS preference, if available."""
    v = str(value).upper().strip()
    if v.startswith("P1"):
        return 1
    if v.startswith("P2"):
        return 2
    if v.startswith("P3"):
        return 3
    if v.startswith("P4"):
        return 4
    if v.startswith("P5"):
        return 5
    if v.startswith("A"):
        return 10
    return 999


def build_variant_table(
    vep_path: str,
    reference_sequence: str,
    strict_labels: bool = False,
    dedupe_mode: str = "id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read VEP table, parse missense variants, and select one row per variant.

    dedupe_mode:
      id       -> one row per #Uploaded_variation, as requested
      mutation -> one row per #Uploaded_variation + protein change
    """
    df = pd.read_csv(vep_path, sep="\t", dtype=str)
    required = {"#Uploaded_variation", "Protein_position", "Amino_acids", "CLIN_SIG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    parsed_records: List[Dict[str, object]] = []
    skipped_records: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        parsed, reason = parse_single_missense_row(row, strict_labels=strict_labels)
        if parsed is None:
            skipped_records.append(
                {
                    "row_index": idx,
                    "variant_id": row.get("#Uploaded_variation", ""),
                    "Protein_position": row.get("Protein_position", ""),
                    "Amino_acids": row.get("Amino_acids", ""),
                    "CLIN_SIG": row.get("CLIN_SIG", ""),
                    "skip_reason": reason,
                }
            )
            continue

        pos0 = parsed.protein_position - 1
        ref_match = 0 <= pos0 < len(reference_sequence) and reference_sequence[pos0] == parsed.ref_aa

        rec = row.to_dict()
        rec.update(
            {
                "row_index": idx,
                "variant_id": parsed.variant_id,
                "clinical_class": parsed.clinical_class,
                "protein_position_int": parsed.protein_position,
                "ref_aa": parsed.ref_aa,
                "alt_aa": parsed.alt_aa,
                "original_CLIN_SIG": parsed.original_label,
                "reference_sequence_aa": reference_sequence[pos0] if 0 <= pos0 < len(reference_sequence) else "OUT_OF_RANGE",
                "matches_reference_sequence": bool(ref_match),
                "mane_rank": 0 if str(row.get("MANE", "")) == "MANE_Select" else 1,
                "mane_select_rank": 0 if str(row.get("MANE_SELECT", "-")) not in {"-", "", "nan"} else 1,
                "tsl_rank": tsl_rank(row.get("TSL", "")),
                "appris_rank": appris_rank(row.get("APPRIS", "")),
            }
        )
        parsed_records.append(rec)

    parsed_df = pd.DataFrame(parsed_records)
    skipped_df = pd.DataFrame(skipped_records)

    if parsed_df.empty:
        raise ValueError("No valid benign/pathogenic single missense variants were parsed.")

    if dedupe_mode == "id":
        parsed_df["dedupe_key"] = parsed_df["variant_id"].astype(str)
    elif dedupe_mode == "mutation":
        parsed_df["dedupe_key"] = (
            parsed_df["variant_id"].astype(str)
            + ":"
            + parsed_df["protein_position_int"].astype(str)
            + parsed_df["ref_aa"].astype(str)
            + ">"
            + parsed_df["alt_aa"].astype(str)
        )
    else:
        raise ValueError("dedupe_mode must be 'id' or 'mutation'")

    warnings_records: List[Dict[str, object]] = []
    selected_rows: List[pd.Series] = []

    for key, group in parsed_df.groupby("dedupe_key", sort=False):
        n_changes = group[["protein_position_int", "ref_aa", "alt_aa"]].drop_duplicates().shape[0]
        n_classes = group["clinical_class"].drop_duplicates().shape[0]
        n_ref_matches = int(group["matches_reference_sequence"].sum())

        if n_changes > 1 or n_classes > 1 or n_ref_matches == 0:
            warnings_records.append(
                {
                    "dedupe_key": key,
                    "n_rows": len(group),
                    "n_distinct_protein_changes": n_changes,
                    "n_distinct_clinical_classes": n_classes,
                    "n_reference_matches": n_ref_matches,
                    "changes_seen": ";".join(
                        sorted(
                            set(
                                group["protein_position_int"].astype(str)
                                + group["ref_aa"].astype(str)
                                + ">"
                                + group["alt_aa"].astype(str)
                            )
                        )
                    ),
                    "classes_seen": ";".join(sorted(set(group["clinical_class"].astype(str)))),
                }
            )

        # Prefer rows consistent with the supplied reference sequence, then MANE, then best TSL/APPRIS.
        group = group.sort_values(
            by=[
                "matches_reference_sequence",
                "mane_rank",
                "mane_select_rank",
                "tsl_rank",
                "appris_rank",
                "row_index",
            ],
            ascending=[False, True, True, True, True, True],
        )
        selected_rows.append(group.iloc[0])

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

    # Skip selected variants that still do not match the supplied reference sequence.
    bad_reference = selected_df[~selected_df["matches_reference_sequence"]].copy()
    if not bad_reference.empty:
        for _, row in bad_reference.iterrows():
            skipped_records.append(
                {
                    "row_index": row.get("row_index", ""),
                    "variant_id": row.get("variant_id", ""),
                    "Protein_position": row.get("Protein_position", ""),
                    "Amino_acids": row.get("Amino_acids", ""),
                    "CLIN_SIG": row.get("CLIN_SIG", ""),
                    "skip_reason": "selected_row_does_not_match_reference_sequence",
                    "reference_sequence_aa": row.get("reference_sequence_aa", ""),
                    "ref_aa": row.get("ref_aa", ""),
                }
            )
        selected_df = selected_df[selected_df["matches_reference_sequence"]].copy()

    skipped_df = pd.DataFrame(skipped_records)
    warnings_df = pd.DataFrame(warnings_records)

    if selected_df.empty:
        raise ValueError(
            "No selected variants match the supplied reference sequence. "
            "Check that your FASTA corresponds to the protein positions in the VEP file."
        )

    return selected_df, skipped_df, warnings_df


def make_centered_window(seq: str, pos_1based: int, max_aa_window: int) -> Tuple[str, int, int, int]:
    """
    Return a sequence window, local 0-based variant index, and absolute start/end.
    max_aa_window counts amino-acid tokens only, not special tokens.
    """
    if not (1 <= pos_1based <= len(seq)):
        raise ValueError(f"Position {pos_1based} is outside sequence length {len(seq)}")

    pos0 = pos_1based - 1
    if len(seq) <= max_aa_window:
        start = 0
        end = len(seq)
    else:
        half = max_aa_window // 2
        start = pos0 - half
        start = max(0, min(start, len(seq) - max_aa_window))
        end = start + max_aa_window

    local_idx = pos0 - start
    return seq[start:end], local_idx, start, end


def aa_token_id(tokenizer, aa: str) -> int:
    tok_id = tokenizer.convert_tokens_to_ids(aa)
    if tok_id is None or tok_id == tokenizer.unk_token_id:
        raise ValueError(f"Could not convert amino acid {aa!r} to a tokenizer ID")
    return int(tok_id)


@torch.inference_mode()
def score_variant(
    model,
    tokenizer,
    sequence: str,
    pos_1based: int,
    ref_aa: str,
    alt_aa: str,
    device: torch.device,
    max_aa_window: int,
) -> Dict[str, object]:
    """Score one substitution using masked-token probabilities."""
    if sequence[pos_1based - 1] != ref_aa:
        raise ValueError(
            f"Reference mismatch at position {pos_1based}: "
            f"FASTA has {sequence[pos_1based - 1]}, VEP has {ref_aa}"
        )

    window_seq, local_idx, window_start, window_end = make_centered_window(
        sequence, pos_1based, max_aa_window=max_aa_window
    )

    masked_tokens = list(window_seq)
    masked_tokens[local_idx] = tokenizer.mask_token
    masked_seq = "".join(masked_tokens)

    encoded = tokenizer(masked_seq, return_tensors="pt", add_special_tokens=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) != 1:
        raise ValueError(f"Expected exactly one mask token, found {len(mask_positions)}")
    mask_index = int(mask_positions.item())

    output = model(**encoded)
    logits = output.logits[0, mask_index]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    ref_id = aa_token_id(tokenizer, ref_aa)
    alt_id = aa_token_id(tokenizer, alt_aa)

    ref_logp = float(log_probs[ref_id].cpu())
    alt_logp = float(log_probs[alt_id].cpu())
    llr = alt_logp - ref_logp

    return {
        "ref_log_probability": ref_logp,
        "alt_log_probability": alt_logp,
        "log_likelihood_ratio": llr,
        "fold_change": math.exp(llr),
        "window_start_1based": window_start + 1,
        "window_end_1based": window_end,
        "local_position_in_window_1based": local_idx + 1,
        "window_length": len(window_seq),
    }


def plot_distributions(scores: pd.DataFrame, out_prefix: str, bins: int = 30) -> None:
    """Create histogram and boxplot for benign/pathogenic ESM log-likelihood ratios.

    The plots are saved to files and not displayed.
    This function avoids Matplotlib-version-dependent boxplot keyword arguments.
    """

    metric = "log_likelihood_ratio"

    class_names = [
        clinical_class
        for clinical_class in ["benign", "pathogenic"]
        if clinical_class in set(scores["clinical_class"])
    ]

    # -------------------------------------------------------------------------
    # Histogram
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 5))

    for clinical_class in class_names:
        values = (
            scores.loc[scores["clinical_class"] == clinical_class, metric]
            .dropna()
            .astype(float)
        )

        if len(values) == 0:
            continue

        plt.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.45,
            label=f"{clinical_class} (n={len(values)})",
        )
        plt.axvline(values.median(), linestyle="--", linewidth=1)

    plt.xlabel("ESM2 log likelihood ratio: log P(mutant) - log P(reference)")
    plt.ylabel("Density")
    plt.title("ESM2-650M missense variant likelihood distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.llr_hist.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Boxplot
    # -------------------------------------------------------------------------
    boxplot_data = []
    boxplot_names = []

    for clinical_class in class_names:
        values = (
            scores.loc[scores["clinical_class"] == clinical_class, metric]
            .dropna()
            .astype(float)
        )

        if len(values) == 0:
            continue

        boxplot_data.append(values)
        boxplot_names.append(clinical_class)

    if len(boxplot_data) == 0:
        warnings.warn("No non-empty clinical classes found for boxplot; skipping boxplot.")
        return

    plt.figure(figsize=(5, 5))

    # Important:
    # Do not pass class names directly to plt.boxplot().
    # Some Matplotlib versions reject the old keyword, and some reject the new one.
    plt.boxplot(boxplot_data, showfliers=True)

    # Set the x-axis class names manually after creating the boxplot.
    plt.xticks(
        ticks=range(1, len(boxplot_names) + 1),
        labels=boxplot_names,
    )

    plt.ylabel("ESM2 log likelihood ratio")
    plt.title("ESM2-650M scores by clinical class")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.llr_boxplot.png", dpi=300)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate VEP missense variants, score with ESM2-650M, and plot benign/pathogenic distributions."
    )
    parser.add_argument("--vep", required=True, help="Input VEP TSV file")
    parser.add_argument("--fasta", required=True, help="Reference protein FASTA")
    parser.add_argument("--out-prefix", default="esm2_missense", help="Output prefix")
    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        help="Hugging Face model name or local model path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc. Default: auto",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 on CUDA to reduce memory use",
    )
    parser.add_argument(
        "--max-aa-window",
        type=int,
        default=1022,
        help="Maximum amino-acid window to send to ESM2. Use <=1022 for standard ESM2 positional length.",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=["id", "mutation"],
        default="id",
        help="id = set of first column (#Uploaded_variation); mutation = keep distinct protein substitutions per ID.",
    )
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="Skip labels containing uncertain/conflicting/not_provided terms.",
    )
    parser.add_argument("--bins", type=int, default=30, help="Histogram bins")
    args = parser.parse_args()

    reference_sequence = read_fasta(args.fasta)
    print(f"Reference sequence length: {len(reference_sequence)} aa")

    variants, skipped, selection_warnings = build_variant_table(
        args.vep,
        reference_sequence,
        strict_labels=args.strict_labels,
        dedupe_mode=args.dedupe_mode,
    )

    print(f"Selected variants for scoring: {len(variants)}")
    print(variants["clinical_class"].value_counts().to_string())

    if not skipped.empty:
        skipped.to_csv(f"{args.out_prefix}.skipped.tsv", sep="\t", index=False)
        print(f"Skipped rows/variants written to: {args.out_prefix}.skipped.tsv")

    if not selection_warnings.empty:
        selection_warnings.to_csv(f"{args.out_prefix}.selection_warnings.tsv", sep="\t", index=False)
        print(f"Selection warnings written to: {args.out_prefix}.selection_warnings.tsv")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.mask_token is None:
        raise ValueError("The selected tokenizer does not have a mask token.")

    load_kwargs = {}
    if device.type == "cuda" and args.fp16:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForMaskedLM.from_pretrained(args.model, **load_kwargs)
    model.to(device)
    model.eval()

    # Keep the amino-acid window safely inside the model's positional limit when available.
    model_max_positions = getattr(model.config, "max_position_embeddings", None)
    if model_max_positions is not None:
        safe_window = max(1, int(model_max_positions) - 2)
        if args.max_aa_window > safe_window:
            print(
                f"Reducing max-aa-window from {args.max_aa_window} to {safe_window} "
                f"based on model.config.max_position_embeddings={model_max_positions}"
            )
            args.max_aa_window = safe_window

    score_records: List[Dict[str, object]] = []
    scoring_skips: List[Dict[str, object]] = []

    for _, row in tqdm(variants.iterrows(), total=len(variants), desc="Scoring variants"):
        try:
            score = score_variant(
                model=model,
                tokenizer=tokenizer,
                sequence=reference_sequence,
                pos_1based=int(row["protein_position_int"]),
                ref_aa=str(row["ref_aa"]),
                alt_aa=str(row["alt_aa"]),
                device=device,
                max_aa_window=int(args.max_aa_window),
            )
            base = {
                "variant_id": row["variant_id"],
                "clinical_class": row["clinical_class"],
                "protein_position": int(row["protein_position_int"]),
                "ref_aa": row["ref_aa"],
                "alt_aa": row["alt_aa"],
                "protein_change": f"{row['ref_aa']}{int(row['protein_position_int'])}{row['alt_aa']}",
                "original_CLIN_SIG": row.get("original_CLIN_SIG", ""),
                "Feature": row.get("Feature", ""),
                "MANE": row.get("MANE", ""),
                "MANE_SELECT": row.get("MANE_SELECT", ""),
                "SYMBOL": row.get("SYMBOL", ""),
            }
            base.update(score)
            score_records.append(base)
        except Exception as exc:
            scoring_skips.append(
                {
                    "variant_id": row.get("variant_id", ""),
                    "protein_position": row.get("protein_position_int", ""),
                    "ref_aa": row.get("ref_aa", ""),
                    "alt_aa": row.get("alt_aa", ""),
                    "skip_reason": f"scoring_error: {exc}",
                }
            )

    scores = pd.DataFrame(score_records)
    if scores.empty:
        raise RuntimeError("No variants were scored successfully.")

    scores.to_csv(f"{args.out_prefix}.scores.tsv", sep="\t", index=False)
    print(f"Scores written to: {args.out_prefix}.scores.tsv")

    if scoring_skips:
        scoring_skips_df = pd.DataFrame(scoring_skips)
        if os.path.exists(f"{args.out_prefix}.skipped.tsv"):
            previous_skips = pd.read_csv(f"{args.out_prefix}.skipped.tsv", sep="\t", dtype=str)
            scoring_skips_df = pd.concat([previous_skips, scoring_skips_df], ignore_index=True)
        scoring_skips_df.to_csv(f"{args.out_prefix}.skipped.tsv", sep="\t", index=False)
        print(f"Additional scoring skips written to: {args.out_prefix}.skipped.tsv")

    summary = scores.groupby("clinical_class")[["log_likelihood_ratio", "fold_change"]].describe()
    summary.to_csv(f"{args.out_prefix}.summary.tsv", sep="\t")
    print(f"Summary written to: {args.out_prefix}.summary.tsv")
    print(summary.to_string())

    plot_distributions(scores, args.out_prefix, bins=args.bins)
    print(f"Plots written to: {args.out_prefix}.llr_hist.png and {args.out_prefix}.llr_boxplot.png")


if __name__ == "__main__":
    main()
