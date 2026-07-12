from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd

from dogma.alphagenome_service import (
    ALPHAGENOME_TRACK_CHOICES,
    SEQUENCE_LENGTH_CHOICES,
)
from dogma.esm_service import ESM_MODEL_CHOICES
from dogma.pipeline import run_dogma_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def run_from_ui(
    api_key: str,
    chromosome: str,
    position: float,
    reference_bases: str,
    alternate_bases: str,
    sequence_length_label: str,
    ontology_text: str,
    selected_tracks: list[str],
    vienna_flank_bp: float,
    gene_symbol_override: str,
    vienna_temperature: float,
    esm_model_checkpoint: str,
    esm_batch_size: float,
    esm_device: str,
    esm_max_sequence_length: float,
    progress: gr.Progress = gr.Progress(),
):
    empty = pd.DataFrame()

    def update_progress(value: float, description: str) -> None:
        progress(value, desc=description)

    try:
        result = run_dogma_pipeline(
            api_key=api_key,
            chromosome=chromosome,
            position=int(position),
            reference_bases=reference_bases,
            alternate_bases=alternate_bases,
            sequence_length_label=sequence_length_label,
            selected_tracks=selected_tracks or [],
            ontology_text=ontology_text,
            vienna_flank_bp=int(vienna_flank_bp),
            gene_symbol_override=gene_symbol_override,
            vienna_temperature=float(vienna_temperature),
            esm_model_checkpoint=esm_model_checkpoint,
            esm_batch_size=int(esm_batch_size),
            esm_device=esm_device,
            esm_max_sequence_length=int(esm_max_sequence_length),
            output_root=OUTPUT_ROOT,
            progress_callback=update_progress,
        )
        return (
            result["status_markdown"],
            result["summary_df"],
            result["alphagenome_df"],
            result["vienna_df"],
            result["isoform_df"],
            result["protein_sequences_df"],
            result["esm_df"],
            result["zip_path"],
        )
    except Exception as exc:
        return (
            f"### DOGMA run failed\n- ❌ {type(exc).__name__}: {exc}",
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            None,
        )


CSS = """
#run-button { min-height: 52px; font-size: 18px; font-weight: 700; }
.sequence-table textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
"""


with gr.Blocks(title="DOGMA: DNA → RNA → Protein") as demo:
    gr.Markdown(
        """
# DOGMA variant pipeline
Enter one **GRCh38** variant. The app runs selected AlphaGenome variant scorers,
folds a strand-aware local RNA window with ViennaRNA, and scores all translated
Ensembl protein isoforms with ESM2 when a complete alternate protein can be reconstructed.

**Prototype scope:** human substitutions/MNVs with equal-length REF and ALT alleles.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. AlphaGenome access")
            api_key = gr.Textbox(
                label="AlphaGenome API key",
                type="password",
                placeholder="Paste key here; it is not written to the output files",
            )
            sequence_length = gr.Dropdown(
                choices=SEQUENCE_LENGTH_CHOICES,
                value="100KB",
                label="AlphaGenome sequence length",
            )
            ontology_text = gr.Textbox(
                value="UBERON:0002046",
                label="Ontology CURIE(s)",
                info="Comma-separated. Leave blank to keep all tracks returned by the selected scorers.",
            )
            selected_tracks = gr.CheckboxGroup(
                choices=ALPHAGENOME_TRACK_CHOICES,
                value=["RNA_SEQ", "ATAC"],
                label="AlphaGenome outputs / recommended scorers",
            )

        with gr.Column(scale=1):
            gr.Markdown("## 2. Variant")
            chromosome = gr.Textbox(value="chr22", label="Chromosome")
            position = gr.Number(value=36201698, precision=0, label="1-based position")
            with gr.Row():
                reference_bases = gr.Textbox(value="A", label="Reference allele")
                alternate_bases = gr.Textbox(value="C", label="Alternate allele")
            gene_symbol_override = gr.Textbox(
                value="",
                label="Gene-symbol override (optional)",
                placeholder="Leave blank to auto-select; e.g. APOL4",
                info=(
                    "Auto-selection uses the strongest protein-coding AlphaGenome gene row, "
                    "then falls back to Ensembl VEP."
                ),
            )

        with gr.Column(scale=1):
            gr.Markdown("## 3. RNA and protein settings")
            vienna_flank_bp = gr.Slider(
                minimum=10,
                maximum=250,
                value=50,
                step=1,
                label="ViennaRNA bases on each side of the variant",
                info="50 gives REF/ALT windows of 101 nt for an SNV.",
            )
            vienna_temperature = gr.Slider(
                minimum=20,
                maximum=45,
                value=37,
                step=0.5,
                label="ViennaRNA temperature (°C)",
            )
            esm_model = gr.Dropdown(
                choices=ESM_MODEL_CHOICES,
                value="esm2_t30_150M_UR50D",
                label="ESM2 model",
            )
            esm_device = gr.Dropdown(
                choices=["cpu", "cuda"],
                value="cpu",
                label="ESM2 device",
                info="Use CPU on an Apple laptop unless your proto_tools build explicitly supports another backend.",
            )
            esm_batch_size = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="ESM2 masked-position batch size",
            )
            esm_max_length = gr.Number(
                value=1000,
                precision=0,
                label="Maximum protein length scored",
                info="Longer isoforms remain in the table but are marked as not scored.",
            )

    run_button = gr.Button("Run complete DOGMA pipeline", variant="primary", elem_id="run-button")
    status = gr.Markdown()
    download = gr.File(label="Download all result tables (.zip)")

    with gr.Tabs():
        with gr.Tab("DOGMA summary"):
            summary_output = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("AlphaGenome scores"):
            alpha_output = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("ViennaRNA scores"):
            vienna_output = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("Ensembl transcript mapping"):
            isoform_output = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("Protein sequences"):
            protein_output = gr.Dataframe(
                interactive=False,
                wrap=True,
                elem_classes=["sequence-table"],
            )
        with gr.Tab("ESM2 scores"):
            esm_output = gr.Dataframe(interactive=False, wrap=True)

    run_button.click(
        fn=run_from_ui,
        inputs=[
            api_key,
            chromosome,
            position,
            reference_bases,
            alternate_bases,
            sequence_length,
            ontology_text,
            selected_tracks,
            vienna_flank_bp,
            gene_symbol_override,
            vienna_temperature,
            esm_model,
            esm_batch_size,
            esm_device,
            esm_max_length,
        ],
        outputs=[
            status,
            summary_output,
            alpha_output,
            vienna_output,
            isoform_output,
            protein_output,
            esm_output,
            download,
        ],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("DOGMA_HOST", "127.0.0.1"),
        server_port=int(os.getenv("DOGMA_PORT", "7860")),
        show_error=True,
        css=CSS,
    )
