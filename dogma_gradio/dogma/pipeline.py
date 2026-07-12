from __future__ import annotations

import json
import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .alphagenome_service import (
    choose_gene_from_alphagenome,
    run_alphagenome_variant_scoring,
)
from .ensembl_service import EnsemblClient
from .esm_service import run_esm_for_isoforms
from .models import VariantInput
from .sequence_utils import parse_csv_text, validate_variant
from .vienna_service import run_vienna_on_genomic_window


ProgressCallback = Callable[[float, str], None]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _notify(callback: ProgressCallback | None, value: float, message: str) -> None:
    if callback is not None:
        callback(value, message)


def run_dogma_pipeline(
    *,
    api_key: str,
    chromosome: str,
    position: int,
    reference_bases: str,
    alternate_bases: str,
    sequence_length_label: str,
    selected_tracks: list[str],
    ontology_text: str,
    vienna_flank_bp: int,
    gene_symbol_override: str,
    vienna_temperature: float,
    esm_model_checkpoint: str,
    esm_batch_size: int,
    esm_device: str,
    esm_max_sequence_length: int,
    output_root: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the three DOGMA modalities while preserving partial results on failure."""
    variant = validate_variant(
        VariantInput(
            chromosome=chromosome,
            position=int(position),
            reference_bases=reference_bases,
            alternate_bases=alternate_bases,
        ),
        substitutions_only=True,
    )
    ontology_curies = parse_csv_text(ontology_text)

    statuses: list[str] = []
    errors: list[str] = []
    alpha_df = _empty_df()
    vienna_df = _empty_df()
    isoform_df = _empty_df()
    protein_sequences_df = _empty_df()
    esm_df = _empty_df()
    summary_df = _empty_df()
    gene_metadata: dict[str, object] = {}

    _notify(progress_callback, 0.05, "Validating the variant with Ensembl VEP")
    ensembl = EnsemblClient(species="homo_sapiens")
    try:
        vep_result = ensembl.annotate_variant(variant)
        vep_candidates = ensembl.candidate_gene_symbols(vep_result)
        statuses.append(
            "Ensembl VEP: success"
            + (f"; protein-coding candidates={', '.join(vep_candidates)}" if vep_candidates else "")
        )
    except Exception as exc:
        vep_result = None
        vep_candidates = []
        errors.append(f"Ensembl VEP failed: {exc}")

    _notify(progress_callback, 0.18, "Running selected AlphaGenome variant scorers")
    try:
        alpha_df = run_alphagenome_variant_scoring(
            api_key=api_key,
            variant=variant,
            sequence_length_label=sequence_length_label,
            selected_tracks=selected_tracks,
            ontology_curies=ontology_curies,
        )
        statuses.append(f"AlphaGenome: success; displayed rows={len(alpha_df):,}")
        if alpha_df.empty and ontology_curies:
            statuses.append(
                "AlphaGenome: no rows remained after ontology filtering; check whether "
                "the chosen output types contain those ontology terms."
            )
    except Exception as exc:
        errors.append(f"AlphaGenome failed: {exc}")
        alpha_df = _empty_df()

    requested_gene = str(gene_symbol_override or "").strip().upper()
    selected_gene = requested_gene or choose_gene_from_alphagenome(alpha_df)
    if not selected_gene and vep_candidates:
        selected_gene = vep_candidates[0].upper()

    if selected_gene:
        statuses.append(f"Protein/RNA gene context selected: {selected_gene}")
    else:
        errors.append(
            "No gene could be selected. Enter a gene-symbol override or include a "
            "gene-linked AlphaGenome output such as RNA_SEQ."
        )

    if selected_gene:
        _notify(progress_callback, 0.42, "Retrieving translated Ensembl isoforms")
        try:
            gene_metadata, isoform_df = ensembl.analyse_gene_isoforms(
                selected_gene,
                variant,
                vep_result=vep_result,
            )
            statuses.append(
                "Ensembl isoforms: success; "
                f"translated transcripts={len(isoform_df):,}; "
                f"unique reference proteins={isoform_df['reference_isoform'].nunique():,}"
            )

            protein_sequences_df = isoform_df[
                [
                    column
                    for column in [
                        "gene_name",
                        "reference_isoform",
                        "alternate_isoform",
                        "transcript_id",
                        "protein_id",
                        "is_ensembl_canonical",
                        "mane_select",
                        "consequence_terms",
                        "alternate_status",
                        "reference_aa_length",
                        "alternate_aa_length",
                        "reference_protein",
                        "alternate_protein",
                    ]
                    if column in isoform_df.columns
                ]
            ].copy()
        except Exception as exc:
            errors.append(f"Ensembl isoform extraction failed: {exc}")
            errors.append(traceback.format_exc(limit=2))
            isoform_df = _empty_df()

    if selected_gene and gene_metadata:
        _notify(progress_callback, 0.60, "Folding strand-aware REF and ALT RNA windows")
        try:
            window = ensembl.extract_genomic_window(variant, int(vienna_flank_bp))
            vienna_df = run_vienna_on_genomic_window(
                window=window,
                gene_name=selected_gene,
                gene_strand=str(gene_metadata["strand"]),
                temperature=float(vienna_temperature),
            )
            statuses.append(
                "ViennaRNA: success; input is a strand-aware genomic/pre-mRNA window"
            )
        except Exception as exc:
            errors.append(f"ViennaRNA failed: {exc}")

    if not isoform_df.empty:
        _notify(progress_callback, 0.75, "Scoring changed protein positions with masked ESM2 inference")
        try:
            esm_df = run_esm_for_isoforms(
                isoform_df,
                model_checkpoint=esm_model_checkpoint,
                batch_size=int(esm_batch_size),
                device=esm_device,
                max_sequence_length=int(esm_max_sequence_length),
            )
            score_column = "delta_position_log_probability_alt_minus_ref"
            resolved = (
                int(esm_df[score_column].notna().sum())
                if not esm_df.empty and score_column in esm_df.columns
                else 0
            )
            statuses.append(
                f"ESM2: success; transcript rows={len(esm_df):,}; "
                f"masked-position ALT-vs-REF comparisons={resolved:,}"
            )
        except Exception as exc:
            errors.append(f"ESM2 failed: {exc}")

    _notify(progress_callback, 0.90, "Saving tables and packaging the run")
    summary_row = {
        "variant": variant.label,
        "selected_gene": selected_gene,
        "gene_strand": gene_metadata.get("strand"),
        "most_severe_vep_consequence": (
            vep_result.get("most_severe_consequence") if isinstance(vep_result, dict) else None
        ),
        "alphagenome_rows": len(alpha_df),
        "vienna_rows": len(vienna_df),
        "translated_transcript_rows": len(isoform_df),
        "unique_reference_protein_isoforms": (
            isoform_df["reference_isoform"].nunique() if not isoform_df.empty else 0
        ),
        "esm_rows": len(esm_df),
        "esm_alt_ref_comparisons": (
            int(esm_df["delta_position_log_probability_alt_minus_ref"].notna().sum())
            if not esm_df.empty
            and "delta_position_log_probability_alt_minus_ref" in esm_df.columns
            else 0
        ),
    }
    summary_df = pd.DataFrame([summary_row])

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    safe_variant = variant.label.replace(":", "_").replace(">", "-")
    job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_variant}_{uuid.uuid4().hex[:8]}"
    output_dir = output_root / job_id
    output_dir.mkdir(parents=True, exist_ok=False)

    tables = {
        "dogma_summary.csv": summary_df,
        "alphagenome_scores.csv": alpha_df,
        "viennarna_scores.csv": vienna_df,
        "ensembl_isoforms.csv": isoform_df,
        "protein_sequences.csv": protein_sequences_df,
        "esm2_scores.csv": esm_df,
    }
    for filename, dataframe in tables.items():
        dataframe.to_csv(output_dir / filename, index=False)

    run_metadata = {
        "variant": variant.label,
        "selected_tracks": selected_tracks,
        "ontology_curies": ontology_curies,
        "selected_gene": selected_gene,
        "gene_metadata": gene_metadata,
        "statuses": statuses,
        "errors": errors,
        "notes": {
            "vienna": (
                "The ViennaRNA input is a transcript-strand-oriented genomic window. "
                "It approximates local pre-mRNA/nascent-RNA folding and is not necessarily "
                "a mature spliced transcript window."
            ),
            "esm": (
                "Alternate proteins are generated only for direct CDS substitutions. "
                "ESM2 compares REF and ALT amino-acid log-probabilities only at masked "
                "changed positions; it does not score every residue in the protein."
            ),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, default=str), encoding="utf-8"
    )

    archive_base = output_root / job_id
    zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_dir))

    status_lines = ["### DOGMA run completed", *[f"- ✅ {item}" for item in statuses]]
    if errors:
        status_lines.extend([f"- ⚠️ {item}" for item in errors])
    status_lines.append(
        "- ViennaRNA interpretation: genomic/pre-mRNA local folding proxy; use transcript "
        "cDNA sequence for mature-mRNA folding."
    )
    status_lines.append(
        "- ESM interpretation: masked-position ALT−REF log-probability compares the "
        "two residues in the same protein context; it is not a pathogenicity probability."
    )

    _notify(progress_callback, 1.0, "Done")
    return {
        "status_markdown": "\n".join(status_lines),
        "summary_df": summary_df,
        "alphagenome_df": alpha_df,
        "vienna_df": vienna_df,
        "isoform_df": isoform_df,
        "protein_sequences_df": protein_sequences_df,
        "esm_df": esm_df,
        "zip_path": str(zip_path),
    }
