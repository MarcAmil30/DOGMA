# ============================================================
# DOGMA Gradio Demo
# AlphaGenome single variant scoring + REF/ALT track plotting
# Google Colab ready
# ============================================================

# ----------------------------
# Cell 1: Install packages
# ----------------------------
from IPython.display import clear_output
clear_output()


# ----------------------------
# Cell 2: Imports
# ----------------------------
import os
import functools
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from alphagenome import colab_utils
from alphagenome.data import genome, gene_annotation, transcript
from alphagenome.models import dna_client, variant_scorers
from alphagenome.visualization import plot_components
from pathlib import Path
import os

OUTPUT_DIR = Path(__file__).resolve().parent / "dogma_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Cell 3: Global config
# ----------------------------

HG38_GTF_FEATHER = (
    "https://storage.googleapis.com/alphagenome/reference/gencode/"
    "hg38/gencode.v46.annotation.gtf.gz.feather"
)

SEQUENCE_LENGTH_OPTIONS = {
    "16KB - fastest": "SEQUENCE_LENGTH_16KB",
    "100KB - good demo default": "SEQUENCE_LENGTH_100KB",
    "500KB - broader context": "SEQUENCE_LENGTH_500KB",
    "1MB - best context but slowest": "SEQUENCE_LENGTH_1MB",
}

PLOT_OUTPUT_MAP = {
    "RNA-seq": dna_client.OutputType.RNA_SEQ,
    "Splice sites": dna_client.OutputType.SPLICE_SITES,
    "Splice site usage": dna_client.OutputType.SPLICE_SITE_USAGE,
    "CAGE": dna_client.OutputType.CAGE,
    "DNase": dna_client.OutputType.DNASE,
    "ATAC": dna_client.OutputType.ATAC,
}

MODEL_CACHE = {}


# ----------------------------
# Cell 4: Helper functions
# ----------------------------

def get_alpha_model(api_key_text):
    """
    Load AlphaGenome model client.
    If API key box is empty, this uses Colab Secrets via colab_utils.get_api_key().
    """
    api_key_text = api_key_text.strip() if api_key_text else ""

    if api_key_text:
        cache_key = "manual_key"
        api_key = api_key_text
    else:
        cache_key = "colab_secret"
        api_key = colab_utils.get_api_key()

    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = dna_client.create(api_key)

    return MODEL_CACHE[cache_key]


@functools.lru_cache(maxsize=1)
def get_transcript_extractor():
    """
    Load GENCODE hg38 transcript annotation once.
    """
    gtf = pd.read_feather(HG38_GTF_FEATHER)

    gtf_tx = gene_annotation.filter_protein_coding(gtf)
    gtf_tx = gene_annotation.filter_transcript_support_level(gtf_tx, ["1"])
    gtf_tx = gene_annotation.filter_to_longest_transcript(gtf_tx)

    return transcript.TranscriptExtractor(gtf_tx)


def clean_scores(df):
    """
    Make score table easier to display.
    """
    df = df.copy()

    for col in ["raw_score", "quantile_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "quantile_score" in df.columns:
        df["abs_quantile_score"] = df["quantile_score"].abs()
    else:
        df["abs_quantile_score"] = None

    if "raw_score" in df.columns:
        df["abs_raw_score"] = df["raw_score"].abs()
    else:
        df["abs_raw_score"] = None

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("")

    return df


def make_summary_table(df_scores):
    summary = (
        df_scores
        .groupby("output_type", dropna=False)
        .agg(
            n_scores=("raw_score", "size"),
            max_abs_quantile=("abs_quantile_score", "max"),
            max_abs_raw_score=("abs_raw_score", "max"),
        )
        .reset_index()
        .sort_values("max_abs_quantile", ascending=False)
    )

    return summary


def make_top_hits_table(df_scores, top_n=30):
    keep_cols = [
        "variant_id",
        "gene_name",
        "gene_type",
        "gene_strand",
        "output_type",
        "variant_scorer",
        "track_name",
        "Assay title",
        "ontology_curie",
        "biosample_name",
        "biosample_type",
        "gtex_tissue",
        "raw_score",
        "quantile_score",
        "abs_quantile_score",
    ]

    keep_cols = [c for c in keep_cols if c in df_scores.columns]

    top_hits = (
        df_scores
        .sort_values("abs_quantile_score", ascending=False)
        [keep_cols]
        .head(top_n)
    )

    return top_hits


def make_score_plot(summary):
    fig, ax = plt.subplots(figsize=(10, 4))

    plot_df = summary.sort_values("max_abs_quantile", ascending=True)

    ax.barh(
        plot_df["output_type"].astype(str),
        plot_df["max_abs_quantile"],
    )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Max absolute quantile score")
    ax.set_ylabel("AlphaGenome output type")
    ax.set_title("DOGMA variant impact summary by modality")

    for i, value in enumerate(plot_df["max_abs_quantile"]):
        ax.text(
            min(value + 0.02, 1.02),
            i,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def add_overlay_track(components, ref_track, alt_track, label):
    """
    Add REF vs ALT overlay if this track exists.
    """
    if ref_track is None or alt_track is None:
        return

    if ref_track.values.shape[-1] == 0:
        return

    components.append(
        plot_components.OverlaidTracks(
            tdata={
                "REF": ref_track,
                "ALT": alt_track,
            },
            colors={
                "REF": "dimgrey",
                "ALT": "red",
            },
            ylabel_template=f"{label}: {{name}} ({{strand}})",
        )
    )


def make_tracks_plot(
    dna_model,
    variant,
    interval,
    ontology_terms,
    selected_plot_outputs,
    strand_filter,
    plot_width,
):
    """
    Predict REF and ALT tracks and return matplotlib figure.
    """
    requested_outputs = [
        PLOT_OUTPUT_MAP[name]
        for name in selected_plot_outputs
        if name in PLOT_OUTPUT_MAP
    ]

    if not requested_outputs:
        requested_outputs = [
            dna_client.OutputType.RNA_SEQ,
            dna_client.OutputType.SPLICE_SITES,
        ]

    variant_tracks = dna_model.predict_variant(
        interval=interval,
        variant=variant,
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs=requested_outputs,
        ontology_terms=ontology_terms,
    )

    ref = variant_tracks.reference
    alt = variant_tracks.alternate

    if strand_filter == "Positive strand only":
        ref = ref.filter_to_strand(strand="+")
        alt = alt.filter_to_strand(strand="+")
    elif strand_filter == "Negative strand only":
        ref = ref.filter_to_strand(strand="-")
        alt = alt.filter_to_strand(strand="-")

    transcript_extractor = get_transcript_extractor()
    transcripts = transcript_extractor.extract(interval)

    components = [
        plot_components.TranscriptAnnotation(transcripts)
    ]

    if "RNA-seq" in selected_plot_outputs:
        add_overlay_track(components, ref.rna_seq, alt.rna_seq, "RNA_SEQ")

    if "Splice sites" in selected_plot_outputs:
        add_overlay_track(components, ref.splice_sites, alt.splice_sites, "SPLICE_SITES")

    if "Splice site usage" in selected_plot_outputs:
        add_overlay_track(
            components,
            ref.splice_site_usage,
            alt.splice_site_usage,
            "SPLICE_SITE_USAGE",
        )

    if "CAGE" in selected_plot_outputs:
        add_overlay_track(components, ref.cage, alt.cage, "CAGE")

    if "DNase" in selected_plot_outputs:
        add_overlay_track(components, ref.dnase, alt.dnase, "DNASE")

    if "ATAC" in selected_plot_outputs:
        add_overlay_track(components, ref.atac, alt.atac, "ATAC")

    plt.close("all")

    plot_components.plot(
        components=components,
        interval=interval.resize(int(plot_width)),
        annotations=[
            plot_components.VariantAnnotation([variant])
        ],
        fig_width=18,
    )

    fig = plt.gcf()
    return fig


def run_dogma(
    api_key_text,
    chromosome,
    position,
    reference_bases,
    alternate_bases,
    sequence_length_choice,
    ontology_terms_text,
    selected_plot_outputs,
    strand_filter,
    plot_width,
    top_n,
):
    """
    Main DOGMA function called by Gradio.
    """
    try:
        dna_model = get_alpha_model(api_key_text)

        chromosome = chromosome.strip()
        position = int(position)
        reference_bases = reference_bases.strip().upper()
        alternate_bases = alternate_bases.strip().upper()

        variant = genome.Variant(
            chromosome=chromosome,
            position=position,
            reference_bases=reference_bases,
            alternate_bases=alternate_bases,
        )

        sequence_length_key = SEQUENCE_LENGTH_OPTIONS[sequence_length_choice]
        sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[sequence_length_key]

        interval = variant.reference_interval.resize(sequence_length)

        status = (
            f"Variant: {variant}\n"
            f"Scored interval: {interval}\n"
            f"Sequence length: {sequence_length_choice}"
        )

        # ----------------------------
        # Score variant
        # ----------------------------
        score_objects = dna_model.score_variant(
            interval=interval,
            variant=variant,
            organism=dna_client.Organism.HOMO_SAPIENS,
            variant_scorers=list(
                variant_scorers.RECOMMENDED_VARIANT_SCORERS.values()
            ),
        )

        df_scores = variant_scorers.tidy_scores(
            score_objects,
            match_gene_strand=True,
        )

        df_scores = clean_scores(df_scores)

        summary = make_summary_table(df_scores)
        top_hits = make_top_hits_table(df_scores, top_n=int(top_n))
        score_plot = make_score_plot(summary)

        # Save full table for download
        safe_variant_name = str(variant).replace(":", "_").replace(">", "_")
        csv_path = OUTPUT_DIR / f"DOGMA_{safe_variant_name}_scores.csv"
        df_scores.to_csv(csv_path, index=False)
        csv_path = str(csv_path)
              

        # ----------------------------
        # Plot tracks
        # ----------------------------
        ontology_terms = [
            x.strip()
            for x in ontology_terms_text.split(",")
            if x.strip()
        ]

        if len(ontology_terms) == 0:
            ontology_terms = ["EFO:0002067"]

        tracks_plot = make_tracks_plot(
            dna_model=dna_model,
            variant=variant,
            interval=interval,
            ontology_terms=ontology_terms,
            selected_plot_outputs=selected_plot_outputs,
            strand_filter=strand_filter,
            plot_width=plot_width,
        )

        status += f"\nNumber of scores: {len(df_scores)}"
        status += f"\nOntology terms for track plot: {ontology_terms}"

        return status, summary, top_hits, score_plot, tracks_plot, csv_path

    except Exception as e:
        error_message = f"Error running DOGMA:\n{type(e).__name__}: {e}"
        empty_df = pd.DataFrame()
        empty_fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "DOGMA failed", ha="center", va="center")
        ax.axis("off")
        return error_message, empty_df, empty_df, empty_fig, empty_fig, None


# ----------------------------
# Cell 5: Gradio UI
# ----------------------------

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

.dogma-card {
    border-radius: 18px;
}
"""

def draw_dna_ui():
    gr.HTML(
        """
        <div id="dogma-title">
            <h1>DOGMA</h1>
            <p>DNA → Omics variant scoring with AlphaGenome</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### Input variant

                Default demo variant:

                `chr22:36201698:A>C`

                Coordinates should be **human hg38**.
                """
            )

            api_key_text = gr.Textbox(
                label="AlphaGenome API key",
                type="password",
                placeholder="Leave empty if using Colab Secrets",
            )

            chromosome = gr.Textbox(
                label="Chromosome",
                value="chr22",
            )

            position = gr.Number(
                label="Position",
                value=36201698,
                precision=0,
            )

            with gr.Row():
                reference_bases = gr.Textbox(
                    label="REF",
                    value="A",
                )
                alternate_bases = gr.Textbox(
                    label="ALT",
                    value="C",
                )

            sequence_length_choice = gr.Dropdown(
                label="Sequence context",
                choices=list(SEQUENCE_LENGTH_OPTIONS.keys()),
                value="100KB - good demo default",
            )

            ontology_terms_text = gr.Textbox(
                label="Ontology terms for track plotting",
                value="EFO:0002067",
                placeholder="Example: EFO:0002067, UBERON:0001157",
            )

            selected_plot_outputs = gr.CheckboxGroup(
                label="Tracks to plot",
                choices=list(PLOT_OUTPUT_MAP.keys()),
                value=[
                    "RNA-seq",
                    "Splice sites",
                    "Splice site usage",
                    "CAGE",
                    "DNase",
                ],
            )

            strand_filter = gr.Radio(
                label="Strand filter",
                choices=[
                    "All strands",
                    "Positive strand only",
                    "Negative strand only",
                ],
                value="All strands",
            )

            plot_width = gr.Slider(
                label="Track plot width around variant",
                minimum=2048,
                maximum=131072,
                step=2048,
                value=32768,
            )

            top_n = gr.Slider(
                label="Number of top hits to show",
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
                lines=6,
            )

            score_plot = gr.Plot(
                label="Score summary visualization",
            )

            tracks_plot = gr.Plot(
                label="REF vs ALT tracks",
            )

    gr.Markdown("## Variant scoring tables")

    with gr.Tab("Summary by modality"):
        summary_table = gr.Dataframe(
            label="Output type summary",
            interactive=False,
            wrap=True,
        )

    with gr.Tab("Top predicted effects"):
        top_hits_table = gr.Dataframe(
            label="Top DOGMA hits",
            interactive=False,
            wrap=True,
        )

    with gr.Tab("Download"):
        csv_download = gr.File(
            label="Download full DOGMA score table",
        )

    run_button.click(
        fn=run_dogma,
        inputs=[
            api_key_text,
            chromosome,
            position,
            reference_bases,
            alternate_bases,
            sequence_length_choice,
            ontology_terms_text,
            selected_plot_outputs,
            strand_filter,
            plot_width,
            top_n,
        ],
        outputs=[
            status_box,
            summary_table,
            top_hits_table,
            score_plot,
            tracks_plot,
            csv_download,
        ],
    )


# ----------------------------
# Cell 6: Launch app
# ----------------------------
if __name__ == "__main__":
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="slate",
        ),
        css=custom_css,
        title="DOGMA",
    ) as demo:
        draw_dna_ui()
    demo.launch(share=True, debug=True)