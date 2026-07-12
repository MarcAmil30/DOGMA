from __future__ import annotations

from typing import Any

import pandas as pd


ESM_MODEL_CHOICES = [
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
]


def _score_dict(score: Any) -> dict[str, float]:
    if isinstance(score, dict):
        return {
            "log_likelihood": float(score["log_likelihood"]),
            "avg_log_likelihood": float(score["avg_log_likelihood"]),
            "perplexity": float(score["perplexity"]),
        }
    return {
        "log_likelihood": float(getattr(score, "log_likelihood")),
        "avg_log_likelihood": float(getattr(score, "avg_log_likelihood")),
        "perplexity": float(getattr(score, "perplexity")),
    }


def run_esm_for_isoforms(
    isoform_df: pd.DataFrame,
    *,
    model_checkpoint: str = "esm2_t30_150M_UR50D",
    batch_size: int = 5,
    device: str = "cpu",
    max_sequence_length: int = 1000,
) -> pd.DataFrame:
    """
    Score reference and directly reconstructable alternate proteins.

    The ESM variant effect is reported as ALT minus REF average log-likelihood.
    Positive values mean the ALT sequence is more likely under ESM2; this is a
    sequence-naturalness comparison, not a direct pathogenicity probability.
    """
    if isoform_df is None or isoform_df.empty:
        return pd.DataFrame()

    try:
        from proto_tools import ESM2ScoringConfig, ESM2ScoringInput, run_esm2_score
    except ImportError as exc:
        raise RuntimeError(
            "proto_tools is not installed in this Python environment. Install the "
            "same proto_tools package/environment used by your working notebooks."
        ) from exc

    max_sequence_length = int(max_sequence_length)
    sequence_order: list[str] = []
    seen: set[str] = set()

    for column in ("reference_protein", "alternate_protein"):
        if column not in isoform_df.columns:
            continue
        for sequence in isoform_df[column].dropna():
            sequence = str(sequence)
            if not sequence or len(sequence) > max_sequence_length or sequence in seen:
                continue
            sequence_order.append(sequence)
            seen.add(sequence)

    score_map: dict[str, dict[str, float]] = {}
    if sequence_order:
        inputs = ESM2ScoringInput(sequences=sequence_order)
        config = ESM2ScoringConfig(
            batch_size=int(batch_size),
            model_checkpoint=str(model_checkpoint),
            return_logits=False,
            device=str(device),
        )
        output = run_esm2_score(inputs, config)
        scores = list(getattr(output, "scores", []))
        if len(scores) != len(sequence_order):
            raise RuntimeError(
                f"ESM2 returned {len(scores)} scores for {len(sequence_order)} sequences."
            )
        score_map = {
            sequence: _score_dict(score)
            for sequence, score in zip(sequence_order, scores, strict=True)
        }

    rows: list[dict[str, Any]] = []
    for _, item in isoform_df.iterrows():
        ref_sequence = item.get("reference_protein")
        alt_sequence = item.get("alternate_protein")
        ref_sequence = str(ref_sequence) if isinstance(ref_sequence, str) else None
        alt_sequence = str(alt_sequence) if isinstance(alt_sequence, str) else None

        ref_score = score_map.get(ref_sequence or "")
        alt_score = score_map.get(alt_sequence or "")

        if ref_sequence and len(ref_sequence) > max_sequence_length:
            esm_status = f"reference_not_scored_length_gt_{max_sequence_length}"
        elif ref_score is None:
            esm_status = "reference_not_scored"
        elif alt_sequence is None:
            esm_status = str(item.get("alternate_status") or "alternate_not_resolved")
        elif len(alt_sequence) > max_sequence_length:
            esm_status = f"alternate_not_scored_length_gt_{max_sequence_length}"
        elif alt_score is None:
            esm_status = "alternate_not_scored"
        else:
            esm_status = "reference_and_alternate_scored"

        row = {
            "gene_name": item.get("gene_name"),
            "reference_isoform": item.get("reference_isoform"),
            "alternate_isoform": item.get("alternate_isoform"),
            "transcript_id": item.get("transcript_id"),
            "protein_id": item.get("protein_id"),
            "is_ensembl_canonical": item.get("is_ensembl_canonical"),
            "consequence_terms": item.get("consequence_terms"),
            "protein_changed": item.get("protein_changed"),
            "alternate_status": item.get("alternate_status"),
            "reference_aa_length": item.get("reference_aa_length"),
            "alternate_aa_length": item.get("alternate_aa_length"),
            "reference_log_likelihood": ref_score["log_likelihood"] if ref_score else None,
            "reference_avg_log_likelihood": (
                ref_score["avg_log_likelihood"] if ref_score else None
            ),
            "reference_perplexity": ref_score["perplexity"] if ref_score else None,
            "alternate_log_likelihood": alt_score["log_likelihood"] if alt_score else None,
            "alternate_avg_log_likelihood": (
                alt_score["avg_log_likelihood"] if alt_score else None
            ),
            "alternate_perplexity": alt_score["perplexity"] if alt_score else None,
            "delta_avg_log_likelihood_alt_minus_ref": (
                alt_score["avg_log_likelihood"] - ref_score["avg_log_likelihood"]
                if ref_score and alt_score
                else None
            ),
            "delta_log_likelihood_alt_minus_ref": (
                alt_score["log_likelihood"] - ref_score["log_likelihood"]
                if ref_score and alt_score
                else None
            ),
            "esm_status": esm_status,
        }
        rows.append(row)

    return pd.DataFrame(rows)
