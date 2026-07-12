from __future__ import annotations

import math
from typing import Any

import pandas as pd


ESM_MODEL_CHOICES = [
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
]

# Column order used by proto_tools ESM2 sampling logits.
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def _log_softmax(logits: list[float]) -> list[float]:
    """Convert one position's logits to numerically stable log-probabilities."""
    values = [float(value) for value in logits]
    maximum = max(values)
    log_denominator = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )
    return [value - log_denominator for value in values]


def _base_row(item: pd.Series) -> dict[str, Any]:
    return {
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
    }


def run_esm_for_isoforms(
    isoform_df: pd.DataFrame,
    *,
    model_checkpoint: str = "esm2_t30_150M_UR50D",
    batch_size: int = 5,
    device: str = "cpu",
    max_sequence_length: int = 1000,
) -> pd.DataFrame:
    """Score only changed amino-acid positions with masked ESM2 inference.

    Each canonical amino-acid substitution is evaluated in its reference protein
    context. The reference residue is replaced by ``_`` and ESM2 is run once for
    that masked context. The result is ALT minus REF log-probability at that
    position; positive values favour ALT and negative values favour REF.

    Multi-residue substitutions produce one row per changed amino acid. Changes
    involving protein length or non-canonical residues (for example a stop) are
    retained as status rows but are not assigned a misleading substitution score.
    """
    if isoform_df is None or isoform_df.empty:
        return pd.DataFrame()

    max_sequence_length = min(int(max_sequence_length), 1022)
    prepared: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for _, item in isoform_df.iterrows():
        base = _base_row(item)
        reference = item.get("reference_protein")
        alternate = item.get("alternate_protein")
        reference = str(reference) if isinstance(reference, str) else None
        alternate = str(alternate) if isinstance(alternate, str) else None

        if reference is None:
            rows.append({**base, "esm_status": "reference_not_available"})
            continue
        if alternate is None:
            rows.append(
                {
                    **base,
                    "esm_status": str(
                        item.get("alternate_status") or "alternate_not_resolved"
                    ),
                }
            )
            continue
        if len(reference) > max_sequence_length:
            rows.append(
                {
                    **base,
                    "esm_status": f"reference_not_scored_length_gt_{max_sequence_length}",
                }
            )
            continue
        if len(reference) != len(alternate):
            rows.append(
                {
                    **base,
                    "esm_status": "not_scored_protein_length_change",
                }
            )
            continue

        changed_positions = [
            index
            for index, (reference_aa, alternate_aa) in enumerate(
                zip(reference, alternate, strict=True)
            )
            if reference_aa != alternate_aa
        ]
        if not changed_positions:
            rows.append({**base, "esm_status": "no_amino_acid_change"})
            continue

        for index in changed_positions:
            reference_aa = reference[index]
            alternate_aa = alternate[index]
            mutation = f"{reference_aa}{index + 1}{alternate_aa}"

            if reference_aa not in AA_ORDER or alternate_aa not in AA_ORDER:
                rows.append(
                    {
                        **base,
                        "mutation": mutation,
                        "mutation_position": index + 1,
                        "reference_aa": reference_aa,
                        "alternate_aa": alternate_aa,
                        "esm_status": "not_scored_noncanonical_or_stop_residue",
                    }
                )
                continue

            masked_sequence = reference[:index] + "_" + reference[index + 1 :]
            prepared.append(
                {
                    **base,
                    "mutation": mutation,
                    "mutation_position": index + 1,
                    "reference_aa": reference_aa,
                    "alternate_aa": alternate_aa,
                    "reference_sequence": reference,
                    "alternate_sequence": alternate,
                    "masked_reference_sequence": masked_sequence,
                }
            )

    if prepared:
        try:
            from proto_tools import ESM2SampleConfig, ESM2SampleInput, run_esm2_sample
        except ImportError as exc:
            raise RuntimeError(
                "proto_tools is not installed in this Python environment. Install the "
                "same proto_tools package/environment used by your working notebooks."
            ) from exc

        # Identical reference context and position need only one model inference.
        unique_masked_sequences = list(
            dict.fromkeys(item["masked_reference_sequence"] for item in prepared)
        )
        output = run_esm2_sample(
            ESM2SampleInput(sequences=unique_masked_sequences),
            ESM2SampleConfig(
                model_checkpoint=str(model_checkpoint),
                batch_size=int(batch_size),
                sampling_method="single_pass",
                return_logits=True,
                device=str(device),
            ),
        )
        logits = getattr(output, "logits", None)
        if logits is None or len(logits) != len(unique_masked_sequences):
            returned = 0 if logits is None else len(logits)
            raise RuntimeError(
                f"ESM2 returned logits for {returned} masked sequence(s); "
                f"expected {len(unique_masked_sequences)}."
            )
        logits_by_sequence = dict(zip(unique_masked_sequences, logits, strict=True))

        for item in prepared:
            position_index = int(item["mutation_position"]) - 1
            sequence_logits = logits_by_sequence[item["masked_reference_sequence"]]
            if position_index >= len(sequence_logits):
                raise RuntimeError(
                    f"ESM2 logits do not contain position {position_index + 1} for "
                    f"{item['transcript_id']}."
                )
            position_logits = sequence_logits[position_index]
            if len(position_logits) != len(AA_ORDER):
                raise RuntimeError(
                    f"ESM2 returned {len(position_logits)} amino-acid logits; "
                    f"expected {len(AA_ORDER)}."
                )

            log_probabilities = _log_softmax(position_logits)
            reference_log_probability = log_probabilities[
                AA_ORDER.index(item["reference_aa"])
            ]
            alternate_log_probability = log_probabilities[
                AA_ORDER.index(item["alternate_aa"])
            ]
            rows.append(
                {
                    **item,
                    "reference_position_log_probability": reference_log_probability,
                    "alternate_position_log_probability": alternate_log_probability,
                    "delta_position_log_probability_alt_minus_ref": (
                        alternate_log_probability - reference_log_probability
                    ),
                    "esm_status": "masked_position_scored",
                }
            )

    return pd.DataFrame(rows)
