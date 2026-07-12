from __future__ import annotations

import os
from typing import Any

import pandas as pd

from .models import VariantInput


ALPHAGENOME_TRACK_CHOICES = [
    "RNA_SEQ",
    "ATAC",
    "DNASE",
    "CAGE",
    "CHIP_HISTONE",
    "CHIP_TF",
    "SPLICE_SITES",
    "SPLICE_SITE_USAGE",
    "SPLICE_JUNCTIONS",
    "CONTACT_MAPS",
    "PROCAP",
]

SEQUENCE_LENGTH_CHOICES = ["2KB", "16KB", "100KB", "500KB", "1MB"]


def _output_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name:
        return str(name).upper()
    text = str(value).upper()
    return text.rsplit(".", 1)[-1]


def _scorer_name(scorer: Any) -> str | None:
    for attr in ("requested_output", "output_type", "requested_outputs"):
        value = getattr(scorer, attr, None)
        if value is not None:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            return _output_name(value)
    return None


def _recommended_scorer_map(variant_scorers: Any, organism: Any) -> dict[str, Any]:
    """Build output-type → scorer mapping across AlphaGenome API versions."""
    mapping: dict[str, Any] = {}

    legacy_mapping = getattr(variant_scorers, "RECOMMENDED_VARIANT_SCORERS", None)
    if isinstance(legacy_mapping, dict):
        mapping.update({str(key).upper(): value for key, value in legacy_mapping.items()})

    get_recommended = getattr(variant_scorers, "get_recommended_scorers", None)
    if callable(get_recommended):
        for scorer in get_recommended(organism):
            name = _scorer_name(scorer)
            if name:
                mapping.setdefault(name, scorer)

    return mapping


def run_alphagenome_variant_scoring(
    *,
    api_key: str,
    variant: VariantInput,
    sequence_length_label: str,
    selected_tracks: list[str],
    ontology_curies: list[str],
) -> pd.DataFrame:
    """Run selected AlphaGenome recommended variant scorers and return tidy scores."""
    api_key = str(api_key or "").strip() or os.getenv("ALPHAGENOME_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "AlphaGenome API key is required in the UI or ALPHAGENOME_API_KEY environment variable."
        )
    if not selected_tracks:
        raise ValueError("Select at least one AlphaGenome track/output type.")

    try:
        from alphagenome.data import genome
        from alphagenome.models import dna_client, variant_scorers
    except ImportError as exc:
        raise RuntimeError(
            "The alphagenome package is not installed. Run: pip install -U alphagenome"
        ) from exc

    organism = dna_client.Organism.HOMO_SAPIENS
    model = dna_client.create(api_key)

    ag_variant = genome.Variant(
        chromosome=variant.chromosome,
        position=int(variant.position),
        reference_bases=variant.reference_bases,
        alternate_bases=variant.alternate_bases,
    )

    sequence_length_label = str(sequence_length_label).upper()
    key = f"SEQUENCE_LENGTH_{sequence_length_label}"
    try:
        sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported AlphaGenome sequence length {sequence_length_label}. "
            f"Choose one of {SEQUENCE_LENGTH_CHOICES}."
        ) from exc

    interval = ag_variant.reference_interval.resize(sequence_length)
    scorer_map = _recommended_scorer_map(variant_scorers, organism)

    selected_names = [str(track).upper() for track in selected_tracks]
    missing = [name for name in selected_names if name not in scorer_map]
    if missing:
        raise ValueError(
            "No recommended AlphaGenome scorer was found for: "
            f"{', '.join(missing)}. Available scorer keys: "
            f"{', '.join(sorted(scorer_map))}"
        )

    scorers = [scorer_map[name] for name in selected_names]
    scores = model.score_variant(
        interval=interval,
        variant=ag_variant,
        variant_scorers=scorers,
        organism=organism,
    )

    try:
        df = variant_scorers.tidy_scores(scores, match_gene_strand=True)
    except TypeError:
        df = variant_scorers.tidy_scores(scores)

    if df is None:
        return pd.DataFrame()
    df = pd.DataFrame(df).copy()

    if "output_type" in df.columns:
        df = df[df["output_type"].astype(str).str.upper().isin(selected_names)].copy()

    # score_variant computes all tracks belonging to the selected scorer(s).
    # ontology filtering is therefore applied to the tidy output table.
    if ontology_curies:
        if "ontology_curie" not in df.columns:
            raise ValueError(
                "AlphaGenome output has no ontology_curie column, so the requested "
                "ontology filter cannot be applied."
            )
        requested = {str(value).strip() for value in ontology_curies if str(value).strip()}
        df = df[df["ontology_curie"].astype(str).isin(requested)].copy()

    if "quantile_score" in df.columns:
        df["abs_quantile_score"] = pd.to_numeric(
            df["quantile_score"], errors="coerce"
        ).abs()
    if "raw_score" in df.columns:
        df["abs_raw_score"] = pd.to_numeric(df["raw_score"], errors="coerce").abs()

    sort_column = "abs_quantile_score" if "abs_quantile_score" in df.columns else "abs_raw_score"
    if sort_column in df.columns:
        df = df.sort_values(sort_column, ascending=False, na_position="last")

    preferred_columns = [
        "gene_id",
        "gene_name",
        "gene_type",
        "gene_strand",
        "output_type",
        "variant_scorer",
        "track_name",
        "track_strand",
        "Assay title",
        "ontology_curie",
        "biosample_name",
        "biosample_type",
        "gtex_tissue",
        "transcription_factor",
        "histone_mark",
        "raw_score",
        "quantile_score",
        "abs_quantile_score",
        "abs_raw_score",
    ]
    visible = [column for column in preferred_columns if column in df.columns]
    remaining = [column for column in df.columns if column not in visible]
    return df[visible + remaining].reset_index(drop=True)


def choose_gene_from_alphagenome(df: pd.DataFrame) -> str | None:
    """Choose the strongest protein-coding gene-linked AlphaGenome row."""
    if df is None or df.empty or "gene_name" not in df.columns:
        return None

    candidates = df[df["gene_name"].notna()].copy()
    candidates = candidates[candidates["gene_name"].astype(str).str.strip() != ""]
    if "gene_type" in candidates.columns:
        protein_coding = candidates[
            candidates["gene_type"].astype(str).str.lower() == "protein_coding"
        ]
        if not protein_coding.empty:
            candidates = protein_coding

    if candidates.empty:
        return None

    for column in ("abs_quantile_score", "abs_raw_score"):
        if column in candidates.columns:
            candidates = candidates.sort_values(column, ascending=False, na_position="last")
            break

    return str(candidates.iloc[0]["gene_name"]).strip()
