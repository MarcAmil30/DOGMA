#!/usr/bin/env python3
"""
ESM2 BRCA1 Missense Variant Dashboard
======================================
A polished Gradio web application that:
  • Runs ESM2-650M missense likelihood scoring with Apple Silicon MPS acceleration
  • Displays an interactive Plotly scatter / violin / bar chart suite
  • Renders the BRCA1 3D structure via py3Dmol embedded in an HTML component,
    coloured by ESM2 log-likelihood ratio
  • Lets users search / filter / download results

Usage:
  python esm2_dashboard.py
  # or with custom paths:
  python esm2_dashboard.py --vep path/to/vep.txt --fasta path/to/seq.fasta
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

# ---------------------------------------------------------------------------
# Default data paths (relative to this script)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
DEFAULT_VEP   = str(_HERE / "45vra2OTHvR2t6dy.Consequence_is_missense_variant.txt")
DEFAULT_FASTA = str(_HERE / "BRCA1_reference.fasta")
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOUR = {
    "benign":     "#4ade80",   # soft green
    "pathogenic": "#f87171",   # soft red
    "bg":         "#0f172a",   # deep navy
    "panel":      "#1e293b",
    "accent":     "#818cf8",   # indigo
    "text":       "#e2e8f0",
}


# ---------------------------------------------------------------------------
# Import the scoring back-end (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_HERE))
from score_esm2_missense_likelihoods import (  # noqa: E402
    get_best_device,
    read_fasta,
    score_all,
)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_STATE: Dict = {
    "scores":   None,   # pd.DataFrame | None
    "skipped":  None,
    "warnings": None,
    "running":  False,
    "log":      [],
    "device":   str(get_best_device()),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device_badge() -> str:
    d = get_best_device()
    icons = {"mps": "🍎 Apple MPS", "cuda": "⚡ CUDA GPU", "cpu": "🖥️ CPU"}
    label = icons.get(d.type, str(d))
    return f"**Active hardware:** `{label}`"


def _clamp_llr(df: pd.DataFrame, col: str = "log_likelihood_ratio") -> pd.DataFrame:
    """Clip extreme LLR values for better visualisation."""
    q01 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    df = df.copy()
    df[col + "_clipped"] = df[col].clip(q01, q99)
    return df


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def fig_scatter(df: pd.DataFrame) -> go.Figure:
    """Interactive scatter: protein position vs LLR, coloured by class."""
    df = _clamp_llr(df)
    colour_map = {"benign": COLOUR["benign"], "pathogenic": COLOUR["pathogenic"]}

    fig = px.scatter(
        df,
        x="protein_position",
        y="log_likelihood_ratio_clipped",
        color="clinical_class",
        color_discrete_map=colour_map,
        hover_data=["variant_id", "protein_change", "log_likelihood_ratio"],
        opacity=0.75,
        title="ESM2 Log-Likelihood Ratio along BRCA1 Sequence",
        labels={
            "protein_position": "Protein Position (aa)",
            "log_likelihood_ratio_clipped": "LLR (clipped p1–p99)",
            "clinical_class": "Clinical Class",
        },
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.4, color="#334155")))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(
        paper_bgcolor=COLOUR["bg"],
        plot_bgcolor=COLOUR["panel"],
        font_color=COLOUR["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def fig_violin(df: pd.DataFrame) -> go.Figure:
    """Violin + strip plot of LLR distributions per clinical class."""
    colour_map = {"benign": COLOUR["benign"], "pathogenic": COLOUR["pathogenic"]}
    fig = go.Figure()
    for cls in ["benign", "pathogenic"]:
        subset = df[df["clinical_class"] == cls]["log_likelihood_ratio"].dropna()
        if subset.empty:
            continue
        col = colour_map[cls]
        fig.add_trace(go.Violin(
            y=subset,
            name=f"{cls} (n={len(subset)})",
            box_visible=True,
            meanline_visible=True,
            fillcolor=col,
            line_color=col,
            opacity=0.7,
            points="outliers",
            marker=dict(color=col, size=3, opacity=0.5),
        ))
    fig.update_layout(
        title="LLR Distribution by Clinical Class",
        yaxis_title="Log-Likelihood Ratio",
        template="plotly_dark",
        paper_bgcolor=COLOUR["bg"],
        plot_bgcolor=COLOUR["panel"],
        font_color=COLOUR["text"],
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def fig_top_variants(df: pd.DataFrame, n: int = 25) -> go.Figure:
    """Horizontal bar chart of the n most extreme variants by |LLR|."""
    df2 = df.copy()
    df2["abs_llr"] = df2["log_likelihood_ratio"].abs()
    top = df2.nlargest(n, "abs_llr").sort_values("log_likelihood_ratio")

    colours = [
        COLOUR["pathogenic"] if row["log_likelihood_ratio"] < 0 else COLOUR["benign"]
        for _, row in top.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=top["log_likelihood_ratio"],
        y=top["protein_change"],
        orientation="h",
        marker_color=colours,
        text=top["clinical_class"],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "LLR: %{x:.3f}<br>"
            "Class: %{text}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"Top {n} Variants by |LLR|",
        xaxis_title="Log-Likelihood Ratio",
        yaxis_title="Variant",
        template="plotly_dark",
        paper_bgcolor=COLOUR["bg"],
        plot_bgcolor=COLOUR["panel"],
        font_color=COLOUR["text"],
        height=max(400, n * 22),
        margin=dict(l=10, r=50, t=40, b=10),
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    return fig


def fig_position_heatmap(df: pd.DataFrame) -> go.Figure:
    """1-D heatmap of mean LLR binned every 50 residues."""
    df2 = df.copy()
    bin_size = 50
    df2["bin"] = (df2["protein_position"] // bin_size) * bin_size
    binned = df2.groupby("bin")["log_likelihood_ratio"].mean().reset_index()
    binned.columns = ["bin_start", "mean_llr"]

    fig = px.bar(
        binned,
        x="bin_start",
        y="mean_llr",
        color="mean_llr",
        color_continuous_scale=["#f87171", "#1e293b", "#4ade80"],
        color_continuous_midpoint=0,
        title="Mean LLR per 50-aa Bin along BRCA1",
        labels={"bin_start": "Residue (bin start)", "mean_llr": "Mean LLR"},
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor=COLOUR["bg"],
        plot_bgcolor=COLOUR["panel"],
        font_color=COLOUR["text"],
        coloraxis_showscale=True,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def fig_summary_stats(df: pd.DataFrame) -> go.Figure:
    """Grouped bar showing mean / median LLR per class."""
    stats = df.groupby("clinical_class")["log_likelihood_ratio"].agg(
        Mean="mean", Median="median"
    ).reset_index()

    fig = go.Figure()
    for stat in ["Mean", "Median"]:
        fig.add_trace(go.Bar(
            name=stat,
            x=stats["clinical_class"],
            y=stats[stat],
            marker_color=[COLOUR["benign"], COLOUR["pathogenic"]][:len(stats)],
            opacity=0.85,
            text=stats[stat].round(3),
            textposition="outside",
        ))
    fig.update_layout(
        title="Summary Statistics",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor=COLOUR["bg"],
        plot_bgcolor=COLOUR["panel"],
        font_color=COLOUR["text"],
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# 3-D structure viewer (py3Dmol via HTML)
# ---------------------------------------------------------------------------

def build_3d_viewer_html(scores_df: Optional[pd.DataFrame] = None) -> str:
    """
    Gradio 6-compatible 3D viewer for BRCA1 (PDB: 1JM7).

    Gradio 6 strips <script> tags from gr.HTML innerHTML, so the full
    self-contained page is embedded inside an <iframe srcdoc="...">,
    where scripts execute in their own document context.
    Residues are coloured by their ESM2 LLR when scores are available.
    """
    import json as _json

    # Build residue colour map from scores
    resi_colours: Dict[str, str] = {}
    if scores_df is not None and not scores_df.empty:
        for _, row in scores_df.iterrows():
            pos = int(row.get("protein_position", 0))
            llr = float(row.get("log_likelihood_ratio", 0))
            if llr < 0:
                t = min(1.0, abs(llr) / 5.0)
                rv, gv, bv = 255, int(100 * (1 - t)), int(100 * (1 - t))
            else:
                t = min(1.0, llr / 5.0)
                rv, gv, bv = int(100 * (1 - t)), int(200 + 55 * t), int(100 * (1 - t))
            resi_colours[str(pos)] = f"#{rv:02x}{gv:02x}{bv:02x}"

    resi_json = _json.dumps(resi_colours)

    inner_html = (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'/>"
        "<style>"
        f"body{{margin:0;background:{COLOUR['bg']};display:flex;flex-direction:column;"
        f"align-items:center;font-family:sans-serif;color:{COLOUR['text']};}}"
        "#viewer{width:100%;height:480px;position:relative;}"
        f".legend{{display:flex;gap:18px;padding:6px 14px;background:{COLOUR['panel']};"
        "border-radius:8px;margin:6px;font-size:12px;}"
        ".dot{width:12px;height:12px;border-radius:50%;display:inline-block;"
        "margin-right:5px;vertical-align:middle;}"
        "</style>"
        "<script src='https://3dmol.org/build/3Dmol-min.js'></script>"
        "</head><body>"
        "<div class='legend'>"
        "<span><span class='dot' style='background:#f87171'></span>Pathogenic / Low LLR</span>"
        "<span><span class='dot' style='background:#e2e8f0'></span>Neutral</span>"
        "<span><span class='dot' style='background:#4ade80'></span>Benign / High LLR</span>"
        "<span><span class='dot' style='background:#93c5fd'></span>No data</span>"
        "</div>"
        "<div id='viewer'></div>"
        "<script>"
        f"const resiColours={resi_json};"
        f"let viewer=$3Dmol.createViewer('viewer',{{backgroundColor:'{COLOUR['bg']}'}});"
        "$3Dmol.download('pdb:1JM7',viewer,{},function(){"
        "viewer.setStyle({},{cartoon:{color:'#93c5fd',opacity:0.85}});"
        "for(const [resi,colour] of Object.entries(resiColours)){"
        "viewer.setStyle({resi:parseInt(resi)},{cartoon:{color:colour,opacity:0.95}});}"
        "viewer.zoomTo();viewer.render();"
        "viewer.setClickable({},true,function(atom){"
        "viewer.removeAllLabels();"
        "if(atom){viewer.addLabel(atom.resn+atom.resi,"
        f"{{position:atom,backgroundColor:'{COLOUR['panel']}',fontColor:'{COLOUR['text']}',fontSize:13,padding:4}});"
        "viewer.render();}});"
        "});"
        "</script></body></html>"
    )

    srcdoc = inner_html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{srcdoc}" '
        'style="width:100%;height:540px;border:none;border-radius:12px;" '
        'sandbox="allow-scripts allow-same-origin"></iframe>'
    )







def _run_scoring(
    vep_path: str,
    fasta_path: str,
    model_name: str,
    max_window: int,
    strict: bool,
    dedupe: str,
) -> None:
    _STATE["running"] = True
    _STATE["log"] = ["🚀 Starting ESM2 scoring pipeline..."]

    def _cb(current, total, msg):
        _STATE["log"].append(f"[{current}/{total}] {msg}")

    try:
        device = get_best_device()
        _STATE["log"].append(f"🔧 Device: {device.type.upper()}")
        _STATE["log"].append(f"📂 VEP:   {vep_path}")
        _STATE["log"].append(f"📂 FASTA: {fasta_path}")
        _STATE["log"].append(f"🤖 Model: {model_name}")
        _STATE["log"].append("⏳ Loading model & tokenizer...")

        scores, skipped, warnings = score_all(
            vep_path=vep_path,
            fasta_path=fasta_path,
            model_name=model_name,
            device=device,
            max_aa_window=max_window,
            strict_labels=strict,
            dedupe_mode=dedupe,
            progress_callback=_cb,
        )

        _STATE["scores"]   = scores
        _STATE["skipped"]  = skipped
        _STATE["warnings"] = warnings
        _STATE["log"].append(f"✅ Done! Scored {len(scores)} variants.")
        n_b = int((scores["clinical_class"] == "benign").sum())
        n_p = int((scores["clinical_class"] == "pathogenic").sum())
        _STATE["log"].append(f"   Benign: {n_b}  |  Pathogenic: {n_p}")
        _STATE["log"].append(f"   Skipped: {len(skipped)}")

    except Exception as exc:
        _STATE["log"].append(f"❌ Error: {exc}")

    finally:
        _STATE["running"] = False


# ---------------------------------------------------------------------------
# Gradio UI callbacks
# ---------------------------------------------------------------------------

def cb_start_scoring(
    vep_path: str,
    fasta_path: str,
    model_name: str,
    max_window: int,
    strict: bool,
    dedupe: str,
) -> str:
    if _STATE["running"]:
        return "⚠️ Scoring already in progress — please wait."
    t = threading.Thread(
        target=_run_scoring,
        args=(vep_path, fasta_path, model_name, max_window, strict, dedupe),
        daemon=True,
    )
    t.start()
    return "🚀 Scoring started — refresh log to track progress."


def cb_poll_log() -> Tuple[str, bool]:
    log_text = "\n".join(_STATE["log"][-60:])  # show last 60 lines
    done = not _STATE["running"] and _STATE["scores"] is not None
    return log_text, done


def cb_load_results(
    llr_min: float,
    llr_max: float,
    class_filter: list,
    search_query: str,
) -> Tuple[
    go.Figure, go.Figure, go.Figure, go.Figure, go.Figure,
    pd.DataFrame, str, str
]:
    df = _STATE["scores"]
    if df is None or df.empty:
        empty = go.Figure()
        return empty, empty, empty, empty, empty, pd.DataFrame(), "No data yet.", ""

    # Filters
    mask = (
        (df["log_likelihood_ratio"] >= llr_min)
        & (df["log_likelihood_ratio"] <= llr_max)
    )
    if class_filter:
        mask &= df["clinical_class"].isin(class_filter)
    if search_query.strip():
        q = search_query.strip().lower()
        mask &= (
            df["variant_id"].str.lower().str.contains(q, na=False)
            | df["protein_change"].str.lower().str.contains(q, na=False)
        )

    filtered = df[mask].copy()

    if filtered.empty:
        empty = go.Figure()
        return empty, empty, empty, empty, empty, pd.DataFrame(), "No variants match filters.", ""

    # Stats card
    n_b = int((filtered["clinical_class"] == "benign").sum())
    n_p = int((filtered["clinical_class"] == "pathogenic").sum())
    mean_b = filtered[filtered["clinical_class"] == "benign"]["log_likelihood_ratio"].mean()
    mean_p = filtered[filtered["clinical_class"] == "pathogenic"]["log_likelihood_ratio"].mean()
    stats_md = (
        f"**Filtered variants:** {len(filtered)}  "
        f"| **Benign:** {n_b} (mean LLR {mean_b:.3f})  "
        f"| **Pathogenic:** {n_p} (mean LLR {mean_p:.3f})"
    )

    display_cols = [
        "variant_id", "protein_change", "clinical_class",
        "log_likelihood_ratio", "fold_change",
        "ref_aa", "alt_aa", "protein_position",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Build 3D viewer HTML
    viewer_html = build_3d_viewer_html(filtered)

    return (
        fig_scatter(filtered),
        fig_violin(filtered),
        fig_top_variants(filtered, n=30),
        fig_position_heatmap(filtered),
        fig_summary_stats(filtered),
        filtered[display_cols].sort_values("log_likelihood_ratio"),
        stats_md,
        viewer_html,
    )


def cb_download_tsv() -> Optional[str]:
    df = _STATE["scores"]
    if df is None or df.empty:
        return None
    out = str(_HERE / "esm2_scores_export.tsv")
    df.to_csv(out, sep="\t", index=False)
    return out


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
  --bg:      {COLOUR['bg']};
  --panel:   {COLOUR['panel']};
  --accent:  {COLOUR['accent']};
  --text:    {COLOUR['text']};
  --green:   {COLOUR['benign']};
  --red:     {COLOUR['pathogenic']};
}}

body, .gradio-container {{
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}}

.gr-button {{
  background: linear-gradient(135deg, var(--accent), #6366f1) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  transition: opacity 0.2s !important;
}}
.gr-button:hover {{ opacity: 0.88 !important; }}

.gr-box, .gr-form, .gr-panel {{
  background: var(--panel) !important;
  border-radius: 12px !important;
  border: 1px solid #334155 !important;
}}

label, .gr-markdown {{
  color: var(--text) !important;
}}

.tab-nav button {{
  color: var(--text) !important;
  font-weight: 500 !important;
}}
.tab-nav button.selected {{
  border-bottom: 2px solid var(--accent) !important;
  color: var(--accent) !important;
}}

textarea, input[type="text"], input[type="number"] {{
  background: #0f172a !important;
  color: var(--text) !important;
  border: 1px solid #334155 !important;
  border-radius: 6px !important;
}}

.hero-banner {{
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1a1a2e 100%);
  border-radius: 16px;
  padding: 28px 36px;
  margin-bottom: 16px;
  border: 1px solid #334155;
}}

.hero-banner h1 {{
  font-size: 2rem;
  font-weight: 700;
  background: linear-gradient(90deg, {COLOUR['accent']}, #a78bfa, {COLOUR['benign']});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 6px 0;
}}

.hero-banner p {{
  color: #94a3b8;
  font-size: 0.95rem;
  margin: 0;
}}

.stat-pill {{
  display: inline-block;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 0.85rem;
  margin: 4px;
  color: var(--text);
}}
"""


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def build_app(vep_path: str = DEFAULT_VEP, fasta_path: str = DEFAULT_FASTA) -> gr.Blocks:
    with gr.Blocks(title="ESM2 BRCA1 Variant Dashboard") as demo:

        # ── Hero banner ──────────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="hero-banner">
          <h1>🧬 ESM2 BRCA1 Missense Variant Dashboard</h1>
          <p>
            Score missense variants with ESM2-650M · Apple Silicon MPS acceleration ·
            Interactive 3D structure · Benign vs pathogenic discrimination
          </p>
          <div style="margin-top:12px">
            <span class="stat-pill">🍎 {_STATE['device'].upper()} backend</span>
            <span class="stat-pill">🤖 ESM2-650M</span>
            <span class="stat-pill">🔬 BRCA1 protein</span>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ================================================================
            # TAB 1 — Run Scoring
            # ================================================================
            with gr.Tab("⚡ Run Scoring"):
                gr.Markdown("### Configure & Launch the Scoring Pipeline")

                with gr.Row():
                    with gr.Column(scale=2):
                        vep_input   = gr.Textbox(label="VEP TSV Path",   value=vep_path,   lines=1)
                        fasta_input = gr.Textbox(label="FASTA Path",      value=fasta_path, lines=1)
                        model_input = gr.Textbox(label="HuggingFace Model", value=DEFAULT_MODEL, lines=1)

                    with gr.Column(scale=1):
                        max_window = gr.Slider(
                            label="Max AA Window", minimum=64, maximum=1022,
                            value=1022, step=1,
                        )
                        strict_chk = gr.Checkbox(label="Strict CLIN_SIG labels", value=False)
                        dedupe_dd  = gr.Dropdown(
                            label="Deduplication mode",
                            choices=["id", "mutation"],
                            value="id",
                        )

                with gr.Row():
                    run_btn     = gr.Button("🚀 Start Scoring", variant="primary", scale=2)
                    refresh_btn = gr.Button("🔄 Refresh Log",   scale=1)

                log_box = gr.Textbox(
                    label="Pipeline Log", lines=18,
                    interactive=False,
                    placeholder="Log output will appear here…",
                )
                done_flag = gr.Checkbox(visible=False, value=False)

                run_btn.click(
                    fn=cb_start_scoring,
                    inputs=[vep_input, fasta_input, model_input, max_window, strict_chk, dedupe_dd],
                    outputs=[log_box],
                )
                refresh_btn.click(
                    fn=lambda: cb_poll_log()[0],
                    inputs=[],
                    outputs=[log_box],
                )

                gr.Markdown(
                    "> **Tip:** After scoring completes, switch to the **📊 Results** tab and click **Apply Filters**."
                )

            # ================================================================
            # TAB 2 — Results & Plots
            # ================================================================
            with gr.Tab("📊 Results & Plots"):
                gr.Markdown("### Filter & Explore Scored Variants")

                with gr.Row():
                    llr_range = gr.Slider(
                        label="LLR Range (min)", minimum=-10, maximum=0,
                        value=-10, step=0.1,
                    )
                    llr_range_max = gr.Slider(
                        label="LLR Range (max)", minimum=0, maximum=10,
                        value=10, step=0.1,
                    )
                    class_dd = gr.CheckboxGroup(
                        label="Clinical Class",
                        choices=["benign", "pathogenic"],
                        value=["benign", "pathogenic"],
                    )
                    search_box = gr.Textbox(
                        label="Search variant / change", placeholder="e.g. R1699Q",
                    )

                filter_btn   = gr.Button("🔍 Apply Filters", variant="primary")
                stats_md_out = gr.Markdown("*Run scoring first, then apply filters.*")

                with gr.Tabs():
                    with gr.Tab("🔵 Scatter"):
                        scatter_plot = gr.Plot(label="Position vs LLR")
                    with gr.Tab("🎻 Violin"):
                        violin_plot = gr.Plot(label="Distribution")
                    with gr.Tab("📊 Top Variants"):
                        top_plot = gr.Plot(label="Extreme Variants")
                    with gr.Tab("🔥 Position Heatmap"):
                        heatmap_plot = gr.Plot(label="Binned Mean LLR")
                    with gr.Tab("📈 Summary Stats"):
                        summary_plot = gr.Plot(label="Mean & Median LLR")

                gr.Markdown("### Variant Table")
                result_table = gr.DataFrame(
                    interactive=False,
                    wrap=True,
                )

                download_btn  = gr.Button("⬇️ Download Full Results as TSV")
                download_file = gr.File(label="Download", visible=False)

                # Hidden viewer HTML placeholder
                viewer_html_state = gr.State("")

                filter_btn.click(
                    fn=cb_load_results,
                    inputs=[llr_range, llr_range_max, class_dd, search_box],
                    outputs=[
                        scatter_plot, violin_plot, top_plot,
                        heatmap_plot, summary_plot,
                        result_table, stats_md_out, viewer_html_state,
                    ],
                )

                download_btn.click(
                    fn=cb_download_tsv,
                    inputs=[],
                    outputs=[download_file],
                )

            # ================================================================
            # TAB 3 — 3D Structure Viewer
            # ================================================================
            with gr.Tab("🧫 3D Structure"):
                gr.Markdown(
                    "### BRCA1 3D Structure coloured by ESM2 LLR\n"
                    "Run scoring and apply filters first. Then click **Render Structure** below."
                )

                render_3d_btn = gr.Button("🎨 Render 3D Structure", variant="primary")
                _ph = (
                    "<div style='padding:40px;text-align:center;color:#94a3b8;"
                    "background:#1e293b;border-radius:12px;font-family:sans-serif'>"
                    "🔬 Click <b>Render 3D Structure</b> above to load the BRCA1 viewer.<br/>"
                    "<small>Requires an internet connection to fetch PDB 1JM7 from RCSB.</small></div>"
                )
                structure_html = gr.HTML(
                    value=_ph,
                    label="3D Viewer",
                )

                def _render_3d():
                    df = _STATE.get("scores")
                    return build_3d_viewer_html(df)

                render_3d_btn.click(
                    fn=_render_3d,
                    inputs=[],
                    outputs=[structure_html],
                )

                gr.Markdown(
                    "> 🍎 **Tip:** The structure loads PDB `1JM7` (BRCA1 BRCT domain) directly "
                    "from the RCSB. Residues in the scored region are coloured: "
                    "🟢 green = high LLR (benign-like) · 🔴 red = low LLR (pathogenic-like) · "
                    "🔵 blue = no variant data."
                )

            # ================================================================
            # TAB 4 — About
            # ================================================================
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
## About this Dashboard

This tool scores **BRCA1 missense variants** using **Meta's ESM2-650M** protein language model,
leveraging Apple Silicon's **Metal Performance Shaders (MPS)** for hardware-accelerated inference.

### Scoring Method
For each variant at position *p*, the residue is **masked** and ESM2 predicts:

| Quantity | Formula |
|----------|---------|
| Log-Likelihood Ratio | `log P(mutant) − log P(reference)` |
| Fold Change | `exp(LLR)` |

A **negative LLR** means the mutant amino acid is *less likely* in context than the wild-type,
which correlates with pathogenicity.

### Hardware Acceleration
```
Device priority: CUDA GPU → Apple MPS → CPU
Current device:  {_STATE['device'].upper()}
```

### Data Sources
- **VEP file:** `{DEFAULT_VEP.split('/')[-1]}`
- **Reference FASTA:** `BRCA1_reference.fasta` (UniProt P38398)
- **3D Structure:** PDB `1JM7` (BRCA1 BRCT domain)

### References
- Rives et al., *PNAS* 2021 – ESM protein language models
- Fraternali lab VEP annotations – BRCA1 missense variant dataset
""")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ESM2 BRCA1 Variant Dashboard")
    parser.add_argument("--vep",   default=DEFAULT_VEP,   help="Default VEP TSV path")
    parser.add_argument("--fasta", default=DEFAULT_FASTA, help="Default FASTA path")
    parser.add_argument("--share", action="store_true",   help="Create a public Gradio link")
    parser.add_argument("--port",  type=int, default=7860, help="Port to serve on")
    args = parser.parse_args()

    # Pass overridden paths directly into build_app() — no global mutation needed.
    app = build_app(vep_path=args.vep, fasta_path=args.fasta)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
        css=_CSS,
    )


if __name__ == "__main__":
    main()
