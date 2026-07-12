# ============================================================
# DOGMA Gradio UI
# Batch AlphaGenome variant scoring
# Filter by ontology_terms + selected output tracks
# Google Colab ready
# ============================================================

# In Google Colab, run this first if needed:
# !pip -q install alphagenome gradio tqdm

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from alphagenome import colab_utils
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers


# ============================================================
# Global settings
# ============================================================

OUTPUT_DIR = Path("dogma_gradio_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CACHE = {}

SEQUENCE_LENGTH_OPTIONS = {
    "16KB - fastest": "SEQUENCE_LENGTH_16KB",
    "100KB - good default": "SEQUENCE_LENGTH_100KB",
    "500KB - broader context": "SEQUENCE_LENGTH_500KB",
    "1MB - slowest / widest context": "SEQUENCE_LENGTH_1MB",
}

TRACK_OPTIONS = {
    "RNA-seq": "RNA_SEQ",
    "CAGE": "CAGE",
    "DNase": "DNASE",
    "ATAC": "ATAC",
    "Splice sites": "SPLICE_SITES",
    "Splice site usage": "SPLICE_SITE_USAGE",
}


# ============================================================
# AlphaGenome client
# ============================================================

def get_alpha_model(api_key_text):
    """
    Load AlphaGenome model client.

    If API key is pasted in the UI, use it.
    If API key box is empty, try Colab Secrets.
    """
    api_key_text = api_key_text.strip() if api_key_text else ""

    if api_key_text:
        cache_key = "manual_key"
        api_key = api_key_text
    else:
        cache_key = "colab_secret"

        try:
            api_key = colab_utils.get_api_key()
        except Exception:
            raise ValueError(
                "No API key found. Paste your AlphaGenome API key in the UI."
            )

    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = dna_client.create(api_key)

    return MODEL_CACHE[cache_key]


# ============================================================
# Variant parsing helpers
# ============================================================

def clean_chromosome(chrom):
    chrom = str(chrom).strip()

    if chrom == "":
        raise ValueError("Chromosome is empty.")

    if not chrom.startswith("chr"):
        chrom = "chr" + chrom

    return chrom


def parse_location(location):
    """
    Parse VEP-style location.

    Examples:
    17:43045712-43045712
    chr17:43045712-43045712
    chr17:43045712
    """
    location = str(location).strip()

    match = re.match(r"^(chr)?([^:]+):(\d+)(?:-(\d+))?$", location)

    if match is None:
        raise ValueError(f"Could not parse Location: {location}")

    chrom = clean_chromosome(match.group(2))
    start = int(match.group(3))
    end = int(match.group(4)) if match.group(4) else start

    return chrom, start, end


def parse_variant_string(s):
    """
    Parse simple variant strings.

    Supported examples:
    chr17:43045712:G>C
    chr17:43045712 G>C
    17:43045712:G>C
    """
    original = str(s).strip()

    if original == "":
        raise ValueError("Empty variant string.")

    cleaned = original.replace(" ", ":")

    match = re.match(
        r"^(chr)?([^:]+):(\d+):?([ACGTacgt]+)>([ACGTacgt]+)$",
        cleaned,
    )

    if match is None:
        raise ValueError(
            f"Could not parse variant string: {original}. "
            "Use format like chr17:43045712:G>C"
        )

    chrom = clean_chromosome(match.group(2))
    position = int(match.group(3))
    ref = match.group(4).upper()
    alt = match.group(5).upper()

    return {
        "Uploaded_variation": original,
        "chromosome": chrom,
        "position": position,
        "end": position,
        "reference_bases": ref,
        "alternate_bases": alt,
    }


def expand_extra_column(df):
    """
    Expand VEP Extra column if present.
    """
    if "Extra" not in df.columns:
        return df

    extra_rows = []

    for x in df["Extra"].fillna(""):
        d = {}

        for item in str(x).split(";"):
            if not item:
                continue

            if "=" in item:
                key, value = item.split("=", 1)
                d[key] = value

        extra_rows.append(d)

    extra_df = pd.DataFrame(extra_rows)

    for col in extra_df.columns:
        if col not in df.columns:
            df[col] = extra_df[col]

    return df


def find_column(df, candidates):
    """
    Find a column using case-insensitive matching.
    """
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def read_any_variant_file(path):
    """
    Reads:
    1. VEP .txt/.tsv files with #Uploaded_variation header
    2. CSV/TSV files with variant columns
    3. Simple text files with one variant per line
    """
    path = Path(path)

    text = path.read_text(errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # VEP-style file
    if any(line.startswith("#Uploaded_variation") for line in lines):
        rows = []
        header = None

        for line in lines:
            if line.startswith("#Uploaded_variation"):
                header = line.lstrip("#").split("\t")
                continue

            if line.startswith("#"):
                continue

            if header is None:
                continue

            parts = line.split("\t")

            if len(parts) < len(header):
                parts = parts + [""] * (len(header) - len(parts))

            rows.append(dict(zip(header, parts[:len(header)])))

        if len(rows) == 0:
            raise ValueError("No VEP rows found in uploaded file.")

        df = pd.DataFrame(rows)

        if "Uploaded_variation" in df.columns:
            df = df[df["Uploaded_variation"] != "Uploaded_variation"].copy()

        df = expand_extra_column(df)

        return df

    # Try CSV/TSV table
    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            dtype=str,
            keep_default_na=False,
        )

        df.columns = [str(c).strip().lstrip("#") for c in df.columns]

        useful_cols = {
            "uploaded_variation",
            "location",
            "allele",
            "ref_allele",
            "chromosome",
            "chrom",
            "chr",
            "position",
            "pos",
            "ref",
            "alt",
            "reference_bases",
            "alternate_bases",
        }

        if any(c.lower() in useful_cols for c in df.columns):
            df = expand_extra_column(df)
            return df

    except Exception:
        pass

    # Fallback: one variant string per line
    return pd.DataFrame({"variant_string": lines})


def finalize_variant_table(df):
    """
    Keep only simple SNVs and collapse duplicate transcript rows.
    """
    df = df.copy()

    valid_bases = {"A", "C", "G", "T"}

    df["reference_bases"] = df["reference_bases"].astype(str).str.upper()
    df["alternate_bases"] = df["alternate_bases"].astype(str).str.upper()

    df["position"] = df["position"].astype(int)
    df["end"] = df["end"].astype(int)

    df["is_simple_snv"] = (
        df["reference_bases"].isin(valid_bases)
        & df["alternate_bases"].isin(valid_bases)
        & (df["position"] == df["end"])
    )

    skipped_df = df[~df["is_simple_snv"]].copy()
    snv_df = df[df["is_simple_snv"]].copy()

    if len(snv_df) == 0:
        raise ValueError(
            "No simple SNVs found. This UI currently supports A/C/G/T SNVs only."
        )

    metadata_cols = [
        "SYMBOL",
        "Gene",
        "Consequence",
        "IMPACT",
        "CLIN_SIG",
        "Existing_variation",
        "SIFT",
        "PolyPhen",
    ]

    metadata_cols = [c for c in metadata_cols if c in snv_df.columns]

    group_cols = [
        "Uploaded_variation",
        "chromosome",
        "position",
        "reference_bases",
        "alternate_bases",
    ]

    agg_dict = {
        c: lambda x: ",".join(sorted(set([str(v) for v in x if str(v) != ""])))
        for c in metadata_cols
    }

    variant_df = (
        snv_df
        .groupby(group_cols, dropna=False)
        .agg(agg_dict)
        .reset_index()
    )

    variant_df["variant_id"] = (
        variant_df["chromosome"].astype(str)
        + ":"
        + variant_df["position"].astype(str)
        + ":"
        + variant_df["reference_bases"].astype(str)
        + ">"
        + variant_df["alternate_bases"].astype(str)
    )

    variant_df["position_mutation"] = (
        variant_df["chromosome"].astype(str)
        + ":"
        + variant_df["position"].astype(str)
        + " "
        + variant_df["reference_bases"].astype(str)
        + ">"
        + variant_df["alternate_bases"].astype(str)
    )

    return variant_df, skipped_df


def make_variant_table_from_df(df):
    """
    Convert uploaded file table into one unique SNV row per variant.
    Supports VEP output or simple variant tables.
    """
    df = df.copy()

    # Case 1: one variant per line
    if "variant_string" in df.columns:
        records = []
        skipped = []

        for s in df["variant_string"]:
            try:
                records.append(parse_variant_string(s))
            except Exception as e:
                skipped.append({"input": s, "error": str(e)})

        variant_input_df = pd.DataFrame(records)

        if len(variant_input_df) == 0:
            raise ValueError("No valid variants found in text file.")

        variant_df, skipped_df = finalize_variant_table(variant_input_df)

        if len(skipped) > 0:
            skipped_df = pd.concat(
                [skipped_df, pd.DataFrame(skipped)],
                ignore_index=True,
            )

        return variant_df, skipped_df

    # Case 2: VEP-style columns
    location_col = find_column(df, ["Location"])
    allele_col = find_column(df, ["Allele"])
    ref_col = find_column(df, ["REF_ALLELE", "ref"])
    uploaded_col = find_column(df, ["Uploaded_variation", "variant", "id"])

    if location_col and allele_col and ref_col:
        parsed = df[location_col].apply(parse_location)

        df["chromosome"] = [x[0] for x in parsed]
        df["position"] = [x[1] for x in parsed]
        df["end"] = [x[2] for x in parsed]

        df["reference_bases"] = df[ref_col].astype(str).str.upper()
        df["alternate_bases"] = df[allele_col].astype(str).str.upper()

        if uploaded_col:
            df["Uploaded_variation"] = df[uploaded_col].astype(str)
        else:
            df["Uploaded_variation"] = ""

        return finalize_variant_table(df)

    # Case 3: simple CSV/TSV columns
    chrom_col = find_column(df, ["chromosome", "chrom", "chr"])
    pos_col = find_column(df, ["position", "pos"])
    ref_col = find_column(df, ["reference_bases", "ref", "reference", "REF_ALLELE"])
    alt_col = find_column(df, ["alternate_bases", "alt", "alternate", "allele"])

    if chrom_col and pos_col and ref_col and alt_col:
        df["chromosome"] = df[chrom_col].apply(clean_chromosome)
        df["position"] = df[pos_col].astype(int)
        df["end"] = df["position"]

        df["reference_bases"] = df[ref_col].astype(str).str.upper()
        df["alternate_bases"] = df[alt_col].astype(str).str.upper()

        uploaded_col = find_column(df, ["Uploaded_variation", "variant", "id"])

        if uploaded_col:
            df["Uploaded_variation"] = df[uploaded_col].astype(str)
        else:
            df["Uploaded_variation"] = (
                df["chromosome"].astype(str)
                + ":"
                + df["position"].astype(str)
                + ":"
                + df["reference_bases"].astype(str)
                + ">"
                + df["alternate_bases"].astype(str)
            )

        return finalize_variant_table(df)

    raise ValueError(
        "Could not understand uploaded file. Use VEP output or columns: "
        "chromosome, position, ref, alt."
    )


def make_manual_variant_table(manual_variant_text, chromosome, position, ref, alt):
    """
    Build one-variant table from UI manual input.
    """
    manual_variant_text = str(manual_variant_text).strip() if manual_variant_text else ""

    if manual_variant_text:
        record = parse_variant_string(manual_variant_text)
    else:
        record = {
            "Uploaded_variation": "manual_variant",
            "chromosome": clean_chromosome(chromosome),
            "position": int(position),
            "end": int(position),
            "reference_bases": str(ref).strip().upper(),
            "alternate_bases": str(alt).strip().upper(),
        }

    df = pd.DataFrame([record])

    return finalize_variant_table(df)


# ============================================================
# Score helpers
# ============================================================

def parse_ontology_terms(ontology_terms_text):
    """
    Parse comma-separated ontology terms.
    Example:
    UBERON:0000310,EFO:0002067
    """
    terms = [
        x.strip()
        for x in str(ontology_terms_text).split(",")
        if x.strip()
    ]

    if len(terms) == 0:
        raise ValueError("Please provide at least one ontology term.")

    return terms


def selected_track_to_output_types(selected_tracks):
    """
    Convert UI tickbox labels to AlphaGenome output_type strings.
    """
    if selected_tracks is None or len(selected_tracks) == 0:
        raise ValueError("Please select at least one track.")

    return [TRACK_OPTIONS[x] for x in selected_tracks]


def clean_scores(df):
    df = df.copy()

    if "raw_score" in df.columns:
        df["raw_score"] = pd.to_numeric(df["raw_score"], errors="coerce")

    if "quantile_score" in df.columns:
        df["quantile_score"] = pd.to_numeric(df["quantile_score"], errors="coerce")
        df["abs_quantile_score"] = df["quantile_score"].abs()
    else:
        raise ValueError("AlphaGenome output did not contain quantile_score.")

    if "raw_score" in df.columns:
        df["abs_raw_score"] = df["raw_score"].abs()

    return df


def score_variants(
    dna_model,
    variant_df,
    sequence_length_choice,
    max_variants,
):
    """
    Score all variants with AlphaGenome.
    """
    sequence_length_key = SEQUENCE_LENGTH_OPTIONS[sequence_length_choice]
    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[sequence_length_key]

    if int(max_variants) > 0:
        variant_df = variant_df.head(int(max_variants)).copy()

    recommended_scorers = list(
        variant_scorers.RECOMMENDED_VARIANT_SCORERS.values()
    )

    all_score_tables = []
    failed_variants = []

    for _, row in variant_df.iterrows():
        try:
            ag_variant = genome.Variant(
                chromosome=row["chromosome"],
                position=int(row["position"]),
                reference_bases=row["reference_bases"],
                alternate_bases=row["alternate_bases"],
            )

            interval = ag_variant.reference_interval.resize(sequence_length)

            score_objects = dna_model.score_variant(
                interval=interval,
                variant=ag_variant,
                organism=dna_client.Organism.HOMO_SAPIENS,
                variant_scorers=recommended_scorers,
            )

            scores = variant_scorers.tidy_scores(
                score_objects,
                match_gene_strand=True,
            )

            scores = clean_scores(scores)

            scores["input_uploaded_variation"] = row["Uploaded_variation"]
            scores["input_variant_id"] = row["variant_id"]
            scores["position_mutation"] = row["position_mutation"]
            scores["input_chromosome"] = row["chromosome"]
            scores["input_position"] = row["position"]
            scores["input_ref"] = row["reference_bases"]
            scores["input_alt"] = row["alternate_bases"]

            for col in [
                "SYMBOL",
                "Gene",
                "Consequence",
                "IMPACT",
                "CLIN_SIG",
                "Existing_variation",
                "SIFT",
                "PolyPhen",
            ]:
                if col in row.index:
                    scores[f"vep_{col}"] = row[col]

            all_score_tables.append(scores)

        except Exception as e:
            failed_variants.append({
                "input_variant_id": row.get("variant_id", ""),
                "position_mutation": row.get("position_mutation", ""),
                "error": f"{type(e).__name__}: {e}",
            })

    if len(all_score_tables) == 0:
        raise ValueError("No variants were scored successfully.")

    all_scores = pd.concat(all_score_tables, ignore_index=True)
    failed_df = pd.DataFrame(failed_variants)

    return all_scores, failed_df, variant_df


def filter_scores_by_ontology_and_tracks(
    all_scores,
    ontology_terms,
    selected_output_types,
):
    """
    Keep rows matching:
    - selected output_type tracks
    - selected ontology_curie terms

    No biosample_name filter is used.
    """
    filtered = all_scores.copy()

    required_cols = ["output_type", "ontology_curie"]

    for col in required_cols:
        if col not in filtered.columns:
            raise ValueError(f"AlphaGenome score table is missing column: {col}")

    filtered["output_type_string"] = filtered["output_type"].astype(str).str.upper()

    selected_output_types_upper = [
        x.upper()
        for x in selected_output_types
    ]

    filtered = filtered[
        filtered["output_type_string"].isin(selected_output_types_upper)
    ].copy()

    filtered = filtered[
        filtered["ontology_curie"].astype(str).isin(ontology_terms)
    ].copy()

    filtered = filtered.drop(columns=["output_type_string"])

    return filtered


def remove_raw_score_columns(df):
    """
    Drop raw score columns before saving/displaying CSVs.
    raw_score is only used for the correlation plot.
    """
    df = df.copy()

    cols_to_drop = ["raw_score", "abs_raw_score"]

    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def make_top_per_variant(filtered_scores):
    """
    Pick one strongest matching score per variant.
    """
    top = (
        filtered_scores
        .sort_values("abs_quantile_score", ascending=False)
        .groupby("input_variant_id", as_index=False)
        .head(1)
        .copy()
    )

    summary_cols = [
        "position_mutation",
        "input_variant_id",
        "input_uploaded_variation",
        "gene_name",
        "output_type",
        "variant_scorer",
        "track_name",
        "Assay title",
        "ontology_curie",
        "biosample_name",
        "biosample_type",
        "gtex_tissue",
        "quantile_score",
        "abs_quantile_score",
        "vep_SYMBOL",
        "vep_Consequence",
        "vep_IMPACT",
        "vep_CLIN_SIG",
        "vep_SIFT",
        "vep_PolyPhen",
    ]

    summary_cols = [c for c in summary_cols if c in top.columns]

    return top[summary_cols]


# ============================================================
# Plot helpers
# ============================================================

def empty_plot(message):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    plt.tight_layout()
    return fig


def make_quantile_plot(top_df, max_plot_variants):
    if len(top_df) == 0:
        return empty_plot("No matching variants to plot.")

    plot_df = (
        top_df
        .sort_values("abs_quantile_score", ascending=False)
        .head(int(max_plot_variants))
        .copy()
    )

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(plot_df)), 5))

    ax.bar(
        plot_df["position_mutation"].astype(str),
        plot_df["quantile_score"],
    )

    ax.axhline(0, linewidth=1)

    ax.set_ylabel("quantile_score")
    ax.set_xlabel("Position mutation")
    ax.set_title("DOGMA: signed effect for selected ontology + tracks")

    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig


def make_abs_quantile_plot(top_df, max_plot_variants):
    if len(top_df) == 0:
        return empty_plot("No matching variants to plot.")

    plot_df = (
        top_df
        .sort_values("abs_quantile_score", ascending=False)
        .head(int(max_plot_variants))
        .copy()
    )

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(plot_df)), 5))

    ax.bar(
        plot_df["position_mutation"].astype(str),
        plot_df["abs_quantile_score"],
    )

    ax.set_ylabel("abs_quantile_score")
    ax.set_xlabel("Position mutation")
    ax.set_title("DOGMA: effect size for selected ontology + tracks")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig


def make_raw_quantile_correlation_plot(filtered_scores):
    """
    Correlation plot: raw_score vs quantile_score.
    raw_score is used only for this plot, not saved in CSVs.
    """
    if "raw_score" not in filtered_scores.columns:
        return empty_plot("raw_score column not available.")

    corr_df = filtered_scores[["raw_score", "quantile_score"]].copy()

    corr_df["raw_score"] = pd.to_numeric(
        corr_df["raw_score"],
        errors="coerce",
    )

    corr_df["quantile_score"] = pd.to_numeric(
        corr_df["quantile_score"],
        errors="coerce",
    )

    corr_df = corr_df.dropna()

    if len(corr_df) < 2:
        return empty_plot(
            "Not enough rows for raw_score vs quantile_score correlation."
        )

    pearson = corr_df["raw_score"].corr(
        corr_df["quantile_score"],
        method="pearson",
    )

    spearman = corr_df["raw_score"].corr(
        corr_df["quantile_score"],
        method="spearman",
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        corr_df["raw_score"],
        corr_df["quantile_score"],
        alpha=0.7,
    )

    if corr_df["raw_score"].nunique() > 1:
        x = corr_df["raw_score"].values
        y = corr_df["quantile_score"].values

        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_line = coef[0] * x_line + coef[1]

        ax.plot(x_line, y_line)

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("raw_score")
    ax.set_ylabel("quantile_score")
    ax.set_title(
        "raw_score vs quantile_score\n"
        f"Pearson={pearson:.3f}, Spearman={spearman:.3f}"
    )

    plt.tight_layout()

    return fig


# ============================================================
# Main Gradio function
# ============================================================

def run_dogma_batch(
    api_key_text,
    manual_variant_text,
    chromosome,
    position,
    ref,
    alt,
    variant_file,
    ontology_terms_text,
    selected_tracks,
    sequence_length_choice,
    max_variants,
    max_plot_variants,
):
    try:
        dna_model = get_alpha_model(api_key_text)

        ontology_terms = parse_ontology_terms(ontology_terms_text)
        selected_output_types = selected_track_to_output_types(selected_tracks)

        # ----------------------------
        # Input: file OR manual variant
        # ----------------------------
        if variant_file is not None:
            input_df = read_any_variant_file(variant_file)
            variant_df, skipped_df = make_variant_table_from_df(input_df)
            input_source = f"Uploaded file: {Path(variant_file).name}"
        else:
            variant_df, skipped_df = make_manual_variant_table(
                manual_variant_text=manual_variant_text,
                chromosome=chromosome,
                position=position,
                ref=ref,
                alt=alt,
            )
            input_source = "Manual variant input"

        # ----------------------------
        # Score variants
        # ----------------------------
        all_scores, failed_df, scored_variant_df = score_variants(
            dna_model=dna_model,
            variant_df=variant_df,
            sequence_length_choice=sequence_length_choice,
            max_variants=max_variants,
        )

        # ----------------------------
        # Filter to ontology terms + selected tracks
        # ----------------------------
        filtered_scores_with_raw = filter_scores_by_ontology_and_tracks(
            all_scores=all_scores,
            ontology_terms=ontology_terms,
            selected_output_types=selected_output_types,
        )

        if len(filtered_scores_with_raw) == 0:
            debug_cols = [
                "output_type",
                "ontology_curie",
                "biosample_name",
                "gtex_tissue",
                "track_name",
            ]

            debug_cols = [
                c for c in debug_cols
                if c in all_scores.columns
            ]

            debug_df = (
                all_scores[debug_cols]
                .drop_duplicates()
                .sort_values(debug_cols)
                .head(100)
            )

            status = (
                "No scores matched the selected ontology terms and tracks.\n\n"
                f"Ontology terms: {ontology_terms}\n"
                f"Selected tracks: {selected_tracks}\n\n"
                "Showing available AlphaGenome score combinations instead."
            )

            return (
                status,
                pd.DataFrame(),
                debug_df,
                empty_plot("No matching scores."),
                empty_plot("No matching scores."),
                empty_plot("No matching scores."),
                None,
                None,
            )

        # ----------------------------
        # Make top summary
        # ----------------------------
        top_per_variant_with_raw = make_top_per_variant(
            filtered_scores_with_raw
        )

        # Remove raw score from tables and CSVs
        filtered_scores_no_raw = remove_raw_score_columns(
            filtered_scores_with_raw
        )

        top_per_variant_no_raw = remove_raw_score_columns(
            top_per_variant_with_raw
        )

        # ----------------------------
        # Plots
        # ----------------------------
        quantile_plot = make_quantile_plot(
            top_per_variant_no_raw,
            max_plot_variants=max_plot_variants,
        )

        abs_quantile_plot = make_abs_quantile_plot(
            top_per_variant_no_raw,
            max_plot_variants=max_plot_variants,
        )

        correlation_plot = make_raw_quantile_correlation_plot(
            filtered_scores_with_raw
        )

        # ----------------------------
        # Save CSVs
        # ----------------------------
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        top_csv_path = OUTPUT_DIR / f"DOGMA_top_per_variant_{run_id}.csv"
        all_csv_path = OUTPUT_DIR / f"DOGMA_all_matching_quantile_scores_{run_id}.csv"
        failed_csv_path = OUTPUT_DIR / f"DOGMA_failed_variants_{run_id}.csv"
        parsed_csv_path = OUTPUT_DIR / f"DOGMA_parsed_unique_variants_{run_id}.csv"
        skipped_csv_path = OUTPUT_DIR / f"DOGMA_skipped_non_snv_rows_{run_id}.csv"

        top_per_variant_no_raw.to_csv(top_csv_path, index=False)
        filtered_scores_no_raw.to_csv(all_csv_path, index=False)
        failed_df.to_csv(failed_csv_path, index=False)
        scored_variant_df.to_csv(parsed_csv_path, index=False)
        skipped_df.to_csv(skipped_csv_path, index=False)

        # ----------------------------
        # UI display tables
        # ----------------------------
        top_display = top_per_variant_no_raw.copy()
        all_display = filtered_scores_no_raw.copy()

        if len(all_display) > 500:
            all_display = all_display.head(500)

        status = (
            "DOGMA run complete.\n\n"
            f"Input source: {input_source}\n"
            f"Unique SNVs parsed: {len(variant_df)}\n"
            f"Variants scored: {scored_variant_df['variant_id'].nunique()}\n"
            f"Failed variants: {len(failed_df)}\n"
            f"Skipped non-SNV rows: {len(skipped_df)}\n\n"
            f"Ontology terms used: {ontology_terms}\n"
            f"Selected tracks: {selected_tracks}\n"
            f"Matching score rows: {len(filtered_scores_no_raw)}\n"
            f"Variants with matching scores: {filtered_scores_no_raw['input_variant_id'].nunique()}\n\n"
            "Note: raw_score is used only for the correlation plot. "
            "Downloaded CSVs do not include raw_score."
        )

        return (
            status,
            top_display,
            all_display,
            quantile_plot,
            abs_quantile_plot,
            correlation_plot,
            str(top_csv_path),
            str(all_csv_path),
        )

    except Exception as e:
        status = f"Error running DOGMA:\n{type(e).__name__}: {e}"

        return (
            status,
            pd.DataFrame(),
            pd.DataFrame(),
            empty_plot("DOGMA failed."),
            empty_plot("DOGMA failed."),
            empty_plot("DOGMA failed."),
            None,
            None,
        )


# ============================================================
# Gradio UI
# ============================================================

custom_css = """
#dogma-title {
    text-align: center;
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(90deg, #111827, #312e81, #4c1d95);
    color: white;
    margin-bottom: 18px;
}

#dogma-title h1 {
    font-size: 42px;
    margin-bottom: 6px;
}

#dogma-title p {
    font-size: 16px;
    opacity: 0.9;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="slate",
    ),
    css=custom_css,
    title="DOGMA",
) as demo:

    gr.HTML(
        """
        <div id="dogma-title">
            <h1>DOGMA</h1>
            <p>DNA variant scoring with AlphaGenome</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### Input

                You can either enter one variant manually or upload a file.

                Manual variant format:

                `chr17:43045712:G>C`

                File options:

                - VEP `.txt`
                - `.csv` / `.tsv` with columns: `chromosome`, `position`, `ref`, `alt`
                - `.txt` with one variant per line
                """
            )

            api_key_text = gr.Textbox(
                label="AlphaGenome API key",
                type="password",
                placeholder="Paste key here, or leave empty if using Colab Secrets",
            )

            manual_variant_text = gr.Textbox(
                label="Manual variant string",
                value="chr17:43045712:G>C",
                placeholder="chr17:43045712:G>C",
            )

            with gr.Row():
                chromosome = gr.Textbox(
                    label="Chromosome",
                    value="chr17",
                )

                position = gr.Number(
                    label="Position",
                    value=43045712,
                    precision=0,
                )

            with gr.Row():
                ref = gr.Textbox(
                    label="REF",
                    value="G",
                )

                alt = gr.Textbox(
                    label="ALT",
                    value="C",
                )

            variant_file = gr.File(
                label="Optional batch file",
                file_types=[".txt", ".tsv", ".csv"],
                type="filepath",
            )

            gr.Markdown("### AlphaGenome filter")

            ontology_terms_text = gr.Textbox(
                label="Ontology terms",
                value="UBERON:0000310",
                placeholder="Example: UBERON:0000310,EFO:0002067",
            )

            selected_tracks = gr.CheckboxGroup(
                label="Tracks to use",
                choices=list(TRACK_OPTIONS.keys()),
                value=["RNA-seq"],
            )

            sequence_length_choice = gr.Dropdown(
                label="Sequence context",
                choices=list(SEQUENCE_LENGTH_OPTIONS.keys()),
                value="100KB - good default",
            )

            max_variants = gr.Slider(
                label="Max variants to score; 0 = all",
                minimum=0,
                maximum=500,
                step=1,
                value=0,
            )

            max_plot_variants = gr.Slider(
                label="Max variants to show in plots",
                minimum=5,
                maximum=100,
                step=5,
                value=30,
            )

            run_button = gr.Button(
                "Run DOGMA",
                variant="primary",
            )

        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="Run status",
                lines=12,
            )

            with gr.Tab("quantile_score plot"):
                quantile_plot = gr.Plot(
                    label="Signed quantile_score",
                )

            with gr.Tab("abs_quantile_score plot"):
                abs_quantile_plot = gr.Plot(
                    label="Absolute quantile_score",
                )

            with gr.Tab("raw_score vs quantile_score"):
                correlation_plot = gr.Plot(
                    label="Correlation plot",
                )

    gr.Markdown("## Results")

    with gr.Tab("Top score per variant"):
        top_table = gr.Dataframe(
            label="One strongest score per variant for selected ontology + tracks",
            interactive=False,
            wrap=True,
        )

    with gr.Tab("All matching quantile scores"):
        all_table = gr.Dataframe(
            label="All matching score rows; first 500 shown",
            interactive=False,
            wrap=True,
        )

    with gr.Tab("Download CSVs"):
        top_csv_download = gr.File(
            label="Download top score per variant CSV",
        )

        all_csv_download = gr.File(
            label="Download all matching quantile scores CSV",
        )

    run_button.click(
        fn=run_dogma_batch,
        inputs=[
            api_key_text,
            manual_variant_text,
            chromosome,
            position,
            ref,
            alt,
            variant_file,
            ontology_terms_text,
            selected_tracks,
            sequence_length_choice,
            max_variants,
            max_plot_variants,
        ],
        outputs=[
            status_box,
            top_table,
            all_table,
            quantile_plot,
            abs_quantile_plot,
            correlation_plot,
            top_csv_download,
            all_csv_download,
        ],
    )


# ============================================================
# Launch
# ============================================================

demo.launch(share=True, debug=True)