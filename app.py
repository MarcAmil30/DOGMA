#!/usr/bin/env python3
"""
Genomics & Transcriptomics Visualizer Dashboard
Author: Antigravity

Multi-tab bioinformatics dashboard built with Gradio.
- DNA tab: AlphaGenome DOGMA variant scoring (alphagenome_UI).
- RNA tab: Static CSV viewer with 3D Plotly + native Matplotlib 2D plotter.
- Protein tab: Placeholder for future development.

Allowed dependencies: gradio, pandas, matplotlib, plotly
"""

import os
import math
import shutil
import tempfile
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import gradio as gr

import sys
from generate_pdbs import (
    parse_dot_bracket as _pdb_parse_dot_bracket,
    generate_folded_coordinates as _pdb_generate_coords,
    write_pdb_structure as _pdb_write_structure,
)
from pyfaidx import Fasta
from Bio.Seq import Seq as BioSeq
from proto_tools import ViennaRNAInput, ViennaRNAConfig, run_viennarna

REF_GENOME_PATH = os.path.join(os.path.dirname(__file__), "reference_genome",
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa")

# ── ESM2 / Protein Analysis back-end ────────────────────────────────────────
import threading
import json as _json
import plotly.express as px
sys.path.insert(0, os.path.dirname(__file__))
from score_esm2_missense_likelihoods import get_best_device, score_all  # noqa: E402

_DEFAULT_VEP   = os.path.join(os.path.dirname(__file__),
                              "45vra2OTHvR2t6dy.Consequence_is_missense_variant.txt")
_DEFAULT_FASTA = os.path.join(os.path.dirname(__file__), "BRCA1_reference.fasta")
_DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

_ESM_STATE = {
    "scores":  None,
    "skipped": None,
    "running": False,
    "log":     [],
}


def _esm_device_label() -> str:
    d = get_best_device()
    return {"mps": "🍎 Apple MPS", "cuda": "⚡ CUDA GPU", "cpu": "🖥️ CPU"}.get(d.type, str(d))


def _esm_run_scoring(vep_path, fasta_path, model_name, max_window, strict, dedupe):
    _ESM_STATE["running"] = True
    _ESM_STATE["log"] = ["🚀 Starting ESM2 scoring pipeline…"]

    def _cb(cur, tot, msg):
        _ESM_STATE["log"].append(f"[{cur}/{tot}] {msg}")

    try:
        device = get_best_device()
        _ESM_STATE["log"].append(f"🔧 Device: {device.type.upper()}")
        _ESM_STATE["log"].append(f"🤖 Model:  {model_name}")
        _ESM_STATE["log"].append("⏳ Loading model & tokenizer (first run may download weights)…")
        scores, skipped, _ = score_all(
            vep_path=vep_path,
            fasta_path=fasta_path,
            model_name=model_name,
            device=device,
            max_aa_window=max_window,
            strict_labels=strict,
            dedupe_mode=dedupe,
            progress_callback=_cb,
        )
        _ESM_STATE["scores"]  = scores
        _ESM_STATE["skipped"] = skipped
        n_b = int((scores["clinical_class"] == "benign").sum())
        n_p = int((scores["clinical_class"] == "pathogenic").sum())
        _ESM_STATE["log"].append(f"✅ Scored {len(scores)} variants — Benign: {n_b} | Pathogenic: {n_p}")
    except Exception as exc:
        _ESM_STATE["log"].append(f"❌ Error: {exc}")
    finally:
        _ESM_STATE["running"] = False


# ─── Plotly helpers (light theme) ───────────────────────────────────────────
_C_BENIGN     = "#2a9d8f"
_C_PATHOGENIC = "#e76f51"
_C_MAP        = {"benign": _C_BENIGN, "pathogenic": _C_PATHOGENIC}
_PLT_THEME    = "plotly_white"
_PLT_LAYOUT   = dict(paper_bgcolor="#ffffff", plot_bgcolor="#f9fafb",
                     font_color="#1f2937", margin=dict(l=10, r=10, t=40, b=10))


def _esm_fig_scatter(df):
    q01, q99 = df["log_likelihood_ratio"].quantile([0.01, 0.99])
    df = df.copy()
    df["llr_c"] = df["log_likelihood_ratio"].clip(q01, q99)
    fig = px.scatter(
        df, x="protein_position", y="llr_c", color="clinical_class",
        color_discrete_map=_C_MAP,
        hover_data=["variant_id", "protein_change", "log_likelihood_ratio"],
        opacity=0.7,
        title="ESM2 LLR along BRCA1",
        labels={"protein_position": "Position (aa)", "llr_c": "LLR (clipped p1–p99)",
                "clinical_class": "Class"},
        template=_PLT_THEME,
    )
    fig.update_traces(marker=dict(size=6))
    fig.add_hline(y=0, line_dash="dot", line_color="#6b7280", line_width=1)
    fig.update_layout(**_PLT_LAYOUT)
    return fig


def _esm_fig_violin(df):
    fig = go.Figure()
    for cls in ["benign", "pathogenic"]:
        vals = df[df["clinical_class"] == cls]["log_likelihood_ratio"].dropna()
        if vals.empty:
            continue
        col = _C_MAP[cls]
        fig.add_trace(go.Violin(
            y=vals, name=f"{cls} (n={len(vals)})",
            box_visible=True, meanline_visible=True,
            fillcolor=col, line_color=col, opacity=0.6,
            points="outliers", marker=dict(color=col, size=3),
        ))
    fig.update_layout(title="LLR Distribution", yaxis_title="Log-Likelihood Ratio",
                      template=_PLT_THEME, **_PLT_LAYOUT)
    return fig


def _esm_fig_top(df, n=25):
    d2 = df.copy()
    d2["abs"] = d2["log_likelihood_ratio"].abs()
    top = d2.nlargest(n, "abs").sort_values("log_likelihood_ratio")
    colours = [_C_PATHOGENIC if v < 0 else _C_BENIGN for v in top["log_likelihood_ratio"]]
    fig = go.Figure(go.Bar(
        x=top["log_likelihood_ratio"], y=top["protein_change"],
        orientation="h", marker_color=colours,
        text=top["clinical_class"], textposition="outside",
        hovertemplate="<b>%{y}</b><br>LLR: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="#6b7280", line_width=1)
    fig.update_layout(title=f"Top {n} Variants by |LLR|",
                      xaxis_title="LLR", yaxis_title="Variant",
                      template=_PLT_THEME, height=max(420, n * 22), **_PLT_LAYOUT)
    return fig


def _esm_fig_heatmap(df):
    d2 = df.copy()
    d2["bin"] = (d2["protein_position"] // 50) * 50
    b = d2.groupby("bin")["log_likelihood_ratio"].mean().reset_index()
    b.columns = ["bin", "mean_llr"]
    fig = px.bar(b, x="bin", y="mean_llr", color="mean_llr",
                 color_continuous_scale=[_C_PATHOGENIC, "#e5e7eb", _C_BENIGN],
                 color_continuous_midpoint=0,
                 title="Mean LLR per 50-aa Bin",
                 labels={"bin": "Residue bin", "mean_llr": "Mean LLR"},
                 template=_PLT_THEME)
    fig.update_layout(**_PLT_LAYOUT)
    return fig


def _esm_fig_summary(df):
    stats = df.groupby("clinical_class")["log_likelihood_ratio"].agg(
        Mean="mean", Median="median").reset_index()
    fig = go.Figure()
    for stat in ["Mean", "Median"]:
        fig.add_trace(go.Bar(
            name=stat, x=stats["clinical_class"], y=stats[stat],
            marker_color=[_C_BENIGN, _C_PATHOGENIC][:len(stats)],
            opacity=0.85, text=stats[stat].round(3), textposition="outside",
        ))
    fig.update_layout(title="Summary Statistics", barmode="group",
                      template=_PLT_THEME, **_PLT_LAYOUT)
    return fig


def _esm_3d_viewer(scores_df=None):
    """Return an iframe-wrapped 3Dmol viewer (light theme). Gradio-6 compatible."""
    resi_colours = {}
    if scores_df is not None and not scores_df.empty:
        for _, row in scores_df.iterrows():
            pos = int(row.get("protein_position", 0))
            llr = float(row.get("log_likelihood_ratio", 0))
            if llr < 0:
                t = min(1.0, abs(llr) / 5.0)
                rv, gv, bv = 255, int(80 * (1 - t)), int(80 * (1 - t))
            else:
                t = min(1.0, llr / 5.0)
                rv, gv, bv = int(42 * (1 - t)), int(157 + 40 * t), int(143 * (1 - t))
            resi_colours[str(pos)] = f"#{rv:02x}{gv:02x}{bv:02x}"

    rj = _json.dumps(resi_colours)
    inner = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        "<style>body{margin:0;background:#f9fafb;display:flex;flex-direction:column;"
        "align-items:center;font-family:Inter,sans-serif;color:#1f2937;}"
        "#v{width:100%;height:460px;position:relative;}"
        ".leg{display:flex;gap:16px;padding:5px 12px;background:#fff;"
        "border:1px solid #e5e7eb;border-radius:6px;margin:6px;font-size:11px;}"
        ".dot{width:11px;height:11px;border-radius:50%;display:inline-block;"
        "margin-right:4px;vertical-align:middle;}</style>"
        "<script src='https://3dmol.org/build/3Dmol-min.js'></script></head><body>"
        "<div class='leg'>"
        "<span><span class='dot' style='background:#e76f51'></span>Pathogenic / Low LLR</span>"
        "<span><span class='dot' style='background:#e5e7eb'></span>Neutral</span>"
        "<span><span class='dot' style='background:#2a9d8f'></span>Benign / High LLR</span>"
        "<span><span class='dot' style='background:#93c5fd'></span>No data</span></div>"
        "<div id='v'></div>"
        "<script>"
        f"const RC={rj};"
        "let vw=$3Dmol.createViewer('v',{backgroundColor:'#f9fafb'});"
        "$3Dmol.download('pdb:1JM7',vw,{},function(){"
        "vw.setStyle({},{cartoon:{color:'#93c5fd',opacity:0.85}});"
        "for(const[r,c]of Object.entries(RC))vw.setStyle({resi:parseInt(r)},{cartoon:{color:c,opacity:0.95}});"
        "vw.zoomTo();vw.render();"
        "vw.setClickable({},true,function(a){"
        "vw.removeAllLabels();"
        "if(a)vw.addLabel(a.resn+a.resi,{position:a,backgroundColor:'#1f2937',fontColor:'#fff',fontSize:12,padding:3});"
        "vw.render();});});"
        "</script></body></html>"
    )
    srcdoc = inner.replace("&", "&amp;").replace('"', "&quot;")
    return (f'<iframe srcdoc="{srcdoc}" '
            'style="width:100%;height:490px;border:1px solid #e5e7eb;border-radius:6px;" '
            'sandbox="allow-scripts allow-same-origin"></iframe>')

# ── End ESM2 helpers ─────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
# 1. DOT-BRACKET PARSER & STRUCTURAL ELEMENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def parse_dot_bracket(structure: str) -> dict:
    """Parse dot-bracket into a base-pair map {i: j, j: i}."""
    bp_map = {}
    stack = []
    for i, ch in enumerate(structure):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                bp_map[i] = j
                bp_map[j] = i
    return bp_map


def classify_elements(structure: str, bp_map: dict) -> list:
    """
    Classify each nucleotide into a structural element.
    Returns a list of element labels, one per nucleotide:
      'stem', 'hairpin', 'interior', 'multiloop', 'exterior'
    """
    n = len(structure)
    labels = ['exterior'] * n

    for i in bp_map:
        labels[i] = 'stem'

    for i in range(n):
        if labels[i] != 'exterior':
            continue

        enclosing_left = -1
        enclosing_right = n
        for j in range(i - 1, -1, -1):
            if j in bp_map and bp_map[j] > i:
                enclosing_left = j
                enclosing_right = bp_map[j]
                break

        if enclosing_left == -1:
            labels[i] = 'exterior'
            continue

        child_stems = 0
        k = enclosing_left + 1
        while k < enclosing_right:
            if k in bp_map and bp_map[k] > k and bp_map[k] < enclosing_right:
                child_stems += 1
                k = bp_map[k] + 1
            else:
                k += 1

        if child_stems == 0:
            labels[i] = 'hairpin'
        elif child_stems == 1:
            labels[i] = 'interior'
        else:
            labels[i] = 'multiloop'

    return labels


# ═══════════════════════════════════════════════════════════════
# 2. NATIVE MATPLOTLIB 2D SECONDARY STRUCTURE PLOTTER
# ═══════════════════════════════════════════════════════════════

ELEMENT_COLORS = {
    'stem':      '#e76f51',
    'hairpin':   '#2a9d8f',
    'interior':  '#457b9d',
    'multiloop': '#e9c46a',
    'exterior':  '#adb5bd',
}

BOND_LEN = 1.0
PAIR_DIST = 0.8


def _layout_coordinates(structure: str, bp_map: dict) -> list:
    """
    Compute 2D (x, y) coordinates for each nucleotide using a
    recursive radial layout algorithm.
    """
    n = len(structure)
    if n == 0:
        return []

    coords = [(0.0, 0.0)] * n

    def layout_region(start, end, origin_x, origin_y, direction_angle):
        if start > end:
            return

        segments = []
        i = start
        while i <= end:
            if i in bp_map and bp_map[i] > i and bp_map[i] <= end:
                j = bp_map[i]
                segments.append(('stem', i, j))
                i = j + 1
            else:
                segments.append(('unpaired', i, i))
                i += 1

        total_items = len(segments)
        if total_items == 0:
            return

        if total_items == 1 and segments[0][0] == 'stem':
            si, sj = segments[0][1], segments[0][2]
            _draw_stem(si, sj, origin_x, origin_y, direction_angle)
            return

        if total_items <= 2:
            radius = BOND_LEN * 1.2
        else:
            radius = max(BOND_LEN * 1.2, (total_items * BOND_LEN) / (2.0 * math.pi))

        cx = origin_x + radius * math.cos(direction_angle)
        cy = origin_y + radius * math.sin(direction_angle)

        start_angle = direction_angle + math.pi
        angle_step = 2.0 * math.pi / max(total_items, 1)

        for idx, seg in enumerate(segments):
            angle = start_angle + idx * angle_step
            px = cx + radius * math.cos(angle)
            py = cy + radius * math.sin(angle)

            if seg[0] == 'unpaired':
                coords[seg[1]] = (px, py)
            else:
                si, sj = seg[1], seg[2]
                _draw_stem(si, sj, px, py, angle)

    def _draw_stem(i, j, ox, oy, angle):
        pairs = []
        ci, cj = i, j
        while ci < cj and ci in bp_map and bp_map[ci] == cj:
            pairs.append((ci, cj))
            ci += 1
            cj -= 1

        perp_angle = angle + math.pi / 2.0

        for step, (pi, pj) in enumerate(pairs):
            ax = ox + step * BOND_LEN * math.cos(angle)
            ay = oy + step * BOND_LEN * math.sin(angle)

            coords[pi] = (ax + PAIR_DIST / 2.0 * math.cos(perp_angle),
                          ay + PAIR_DIST / 2.0 * math.sin(perp_angle))
            coords[pj] = (ax - PAIR_DIST / 2.0 * math.cos(perp_angle),
                          ay - PAIR_DIST / 2.0 * math.sin(perp_angle))

        if ci <= cj:
            last_ax = ox + len(pairs) * BOND_LEN * math.cos(angle)
            last_ay = oy + len(pairs) * BOND_LEN * math.sin(angle)
            layout_region(ci, cj, last_ax, last_ay, angle)

    # Top-level layout
    segments = []
    i = 0
    while i < n:
        if i in bp_map and bp_map[i] > i:
            j = bp_map[i]
            segments.append(('stem', i, j))
            i = j + 1
        else:
            segments.append(('unpaired', i, i))
            i += 1

    cursor_x = 0.0
    cursor_y = 0.0
    base_angle = math.pi / 2.0

    for seg in segments:
        if seg[0] == 'unpaired':
            coords[seg[1]] = (cursor_x, cursor_y)
            cursor_x += BOND_LEN
        else:
            si, sj = seg[1], seg[2]
            _draw_stem(si, sj, cursor_x, cursor_y, base_angle)
            cursor_x += BOND_LEN * 1.5

    return coords


def generate_2d_plot(sequence: str, structure: str):
    """
    Generate a Matplotlib figure of the 2D secondary structure.
    Color-coded by structural element. White background, no axes.
    """
    plt.close("all")
    try:
        n = len(structure)
        if n == 0 or len(sequence) != n:
            raise ValueError(f"Sequence length ({len(sequence)}) != structure length ({n})")

        bp_map = parse_dot_bracket(structure)
        labels = classify_elements(structure, bp_map)
        coords = _layout_coordinates(structure, bp_map)

        if not coords or len(coords) != n:
            raise ValueError("Layout engine produced invalid coordinates")

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
        ax.set_facecolor('white')

        # Backbone
        ax.plot(xs, ys, '-', color='#6b7280', linewidth=1.2, alpha=0.5, zorder=1)

        # Base pairs
        for i, j in bp_map.items():
            if i < j:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                        '-', color='#9ca3af', linewidth=0.8, alpha=0.6, zorder=2)

        # Nucleotide dots
        for idx in range(n):
            color = ELEMENT_COLORS.get(labels[idx], '#adb5bd')
            ax.scatter(xs[idx], ys[idx], c=color, s=40, zorder=3,
                       edgecolors='#374151', linewidths=0.4)

        # Labels for short sequences
        if n <= 120:
            for idx in range(n):
                ax.annotate(sequence[idx],
                            (xs[idx], ys[idx]),
                            fontsize=max(4, min(7, 600 // n)),
                            ha='center', va='center',
                            color='#1f2937', fontweight='bold', zorder=4)

        legend_patches = [
            mpatches.Patch(color=ELEMENT_COLORS['stem'],      label='Stem'),
            mpatches.Patch(color=ELEMENT_COLORS['hairpin'],    label='Hairpin Loop'),
            mpatches.Patch(color=ELEMENT_COLORS['interior'],   label='Interior Loop / Bulge'),
            mpatches.Patch(color=ELEMENT_COLORS['multiloop'],  label='Multiloop Junction'),
            mpatches.Patch(color=ELEMENT_COLORS['exterior'],   label='Exterior / Dangling'),
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=7,
                  frameon=True, facecolor='white', edgecolor='#e5e7eb',
                  framealpha=0.9)

        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        return fig

    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
        ax.set_facecolor('white')
        ax.text(0.5, 0.5, f"2D Layout Error:\n{str(e)}",
                ha='center', va='center', color='#374151',
                fontsize=11, fontstyle='italic',
                transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        return fig


# ═══════════════════════════════════════════════════════════════
# 3. 3D COARSE-GRAINED PLOTLY VIEWER (Force-Directed Layout)
# ═══════════════════════════════════════════════════════════════

def generate_folded_coordinates(sequence: str, bp_map: dict, iterations: int = 80) -> np.ndarray:
    """Numpy-vectorized force-directed 3D layout for RNA backbone."""
    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3))

    angles = 2.0 * np.pi * np.arange(n) / max(10, n * 0.3)
    r = 8.0 + n * 0.05
    z_vals = (np.arange(n) - n / 2) * 1.5
    coords = np.column_stack([
        r * np.cos(angles) + np.random.uniform(-0.1, 0.1, n),
        r * np.sin(angles) + np.random.uniform(-0.1, 0.1, n),
        z_vals + np.random.uniform(-0.1, 0.1, n),
    ])

    bp_i = np.array([i for i, j in bp_map.items() if i < j], dtype=int)
    bp_j = np.array([j for i, j in bp_map.items() if i < j], dtype=int)
    chain_i = np.arange(n - 1)
    chain_j = np.arange(1, n)

    dt = 0.15
    for _ in range(iterations):
        forces = np.zeros_like(coords)

        d = coords[chain_j] - coords[chain_i]
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        dist = np.maximum(dist, 0.01)
        f = d / dist * (dist - 4.0) * 0.8
        np.add.at(forces, chain_i, f)
        np.add.at(forces, chain_j, -f)

        if len(bp_i) > 0:
            d = coords[bp_j] - coords[bp_i]
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            dist = np.maximum(dist, 0.01)
            f = d / dist * (dist - 2.8) * 1.5
            np.add.at(forces, bp_i, f)
            np.add.at(forces, bp_j, -f)

        for i in range(n):
            d = coords[i + 1:] - coords[i]
            dist = np.linalg.norm(d, axis=1)
            dist = np.maximum(dist, 0.01)
            mask = dist < 6.0
            if i + 1 < n:
                mask[0] = False
            if i in bp_map:
                partner = bp_map[i]
                if partner > i:
                    idx = partner - i - 1
                    if idx < len(mask):
                        mask[idx] = False
            if not np.any(mask):
                continue
            repel = (6.0 - dist[mask])[:, None]
            d_unit = d[mask] / dist[mask, None]
            f = d_unit * repel * 0.4
            forces[i] -= f.sum(axis=0)
            indices = np.where(mask)[0] + i + 1
            np.add.at(forces, indices, f)

        forces -= coords * 0.01

        f_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        clamped = f_norm > 4.0
        if np.any(clamped):
            forces = np.where(clamped, forces / f_norm * 4.0, forces)
        coords += forces * dt

    return coords


def generate_3d_plotly(sequence: str, structure: str, mfe: float):
    """Build a Plotly 3D scatter figure for the RNA. White background."""
    fig = go.Figure()
    bp_map = parse_dot_bracket(structure)
    n = len(sequence)

    if n == 0:
        fig.update_layout(title="Empty sequence", template='plotly_white')
        return fig

    try:
        coords = generate_folded_coordinates(sequence, bp_map, iterations=80)
    except Exception:
        fig.update_layout(title="3D layout error", template='plotly_white')
        return fig

    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='lines', name='Backbone',
        line=dict(color='#6b7280', width=2.5, dash='dash'),
        hoverinfo='skip',
    ))

    bp_x, bp_y, bp_z = [], [], []
    for i, j in bp_map.items():
        if i < j and i < n and j < n:
            bp_x.extend([coords[i][0], coords[j][0], None])
            bp_y.extend([coords[i][1], coords[j][1], None])
            bp_z.extend([coords[i][2], coords[j][2], None])
    if bp_x:
        fig.add_trace(go.Scatter3d(
            x=bp_x, y=bp_y, z=bp_z,
            mode='lines', name='Base Pairs',
            line=dict(color='#a78bfa', width=2, dash='dash'),
            hoverinfo='skip',
        ))

    nt_colors = {'A': '#ef4444', 'U': '#3b82f6', 'G': '#22c55e', 'C': '#eab308'}
    for nt, color in nt_colors.items():
        indices = [idx for idx in range(n) if sequence[idx].upper() == nt]
        if indices:
            nt_coords = coords[indices]
            hover = [f"Residue {idx+1}: {nt}<br>{'Paired' if idx in bp_map else 'Unpaired'}"
                     for idx in indices]
            fig.add_trace(go.Scatter3d(
                x=nt_coords[:, 0], y=nt_coords[:, 1], z=nt_coords[:, 2],
                mode='markers', name=nt,
                marker=dict(size=6, color=color, opacity=0.9,
                            line=dict(color='#e5e7eb', width=0.5)),
                hovertext=hover, hoverinfo='text',
            ))

    axis_cfg = dict(
        showgrid=False, zeroline=False, showline=False,
        showbackground=True, backgroundcolor='white',
        ticks='', showticklabels=False, title='',
    )
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='white', plot_bgcolor='white',
        scene=dict(xaxis=axis_cfg, yaxis=axis_cfg, zaxis=axis_cfg,
                   bgcolor='white', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=10),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#e5e7eb', borderwidth=1,
                    font=dict(color='#1f2937', size=11)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# 4. CALLBACK HANDLERS
# ═══════════════════════════════════════════════════════════════

def handle_csv_upload(file):
    """Parse uploaded CSV and return the selector dataframe."""
    empty = pd.DataFrame(columns=["Index", "Sequence", "Structure", "MFE"])

    if file is None:
        return None, empty, None, None, "Upload a CSV with columns: **sequence**, **structure**, **mfe**."

    try:
        df = pd.read_csv(file.name)
        df.columns = [c.strip().lower() for c in df.columns]

        required = {'sequence', 'structure', 'mfe'}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            return None, empty, None, None, f"❌ Missing columns: {', '.join(missing)}."

        selector_data = []
        for i, row in df.iterrows():
            seq = str(row['sequence']).strip()
            struct = str(row['structure']).strip()
            mfe_val = float(row['mfe'])
            selector_data.append({
                "Index": i,
                "Sequence": seq[:60] + "…" if len(seq) > 60 else seq,
                "Structure": struct[:60] + "…" if len(struct) > 60 else struct,
                "MFE": round(mfe_val, 4),
            })

        selector_df = pd.DataFrame(selector_data)

        row0 = df.iloc[0]
        seq0 = str(row0['sequence']).strip()
        struct0 = str(row0['structure']).strip()
        mfe0 = float(row0['mfe'])

        return df, selector_df, generate_3d_plotly(seq0, struct0, mfe0), generate_2d_plot(seq0, struct0), _build_metadata(seq0, struct0, mfe0, 0)

    except Exception as e:
        return None, empty, None, None, f"❌ Error parsing CSV: {str(e)}"


def handle_row_select(evt: gr.SelectData, df):
    """When user clicks a row in the selector table, update all visuals."""
    if df is None:
        return None, None, "No data loaded."

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_idx >= len(df):
        return None, None, f"Row index {row_idx} out of range."

    row = df.iloc[row_idx]
    seq = str(row['sequence']).strip()
    struct = str(row['structure']).strip()
    mfe_val = float(row['mfe'])

    return generate_3d_plotly(seq, struct, mfe_val), generate_2d_plot(seq, struct), _build_metadata(seq, struct, mfe_val, row_idx)


def handle_generate_pdbs(df):
    """
    Generate a .pdb file for every row in the loaded DataFrame using
    the force-directed 3D model from generate_pdbs.py, zip them, and
    return the zip path for download.
    """
    if df is None or len(df) == 0:
        return None, "❌ No CSV loaded. Upload a CSV file first."

    try:
        # Write PDBs into a temp directory
        tmp_dir = tempfile.mkdtemp(prefix="rna_pdbs_")
        success = 0
        errors = []

        for idx, row in df.iterrows():
            seq = str(row['sequence']).strip()
            struct = str(row['structure']).strip()

            if not seq or not struct or len(seq) != len(struct):
                errors.append(f"Row {idx}: length mismatch or empty — skipped.")
                continue

            filename = f"structure_{idx}.pdb"
            filepath = os.path.join(tmp_dir, filename)

            try:
                bp_map = _pdb_parse_dot_bracket(struct)
                coords = _pdb_generate_coords(seq, bp_map, iterations=150)
                _pdb_write_structure(filepath, seq, bp_map, coords)
                success += 1
            except Exception as e:
                errors.append(f"Row {idx}: {e}")

        if success == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None, f"❌ No PDB files generated. Errors:\n" + "\n".join(errors[:5])

        # Zip the folder
        zip_path = tmp_dir + ".zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname in sorted(os.listdir(tmp_dir)):
                zf.write(os.path.join(tmp_dir, fname), arcname=fname)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        err_summary = f"  ({len(errors)} skipped)" if errors else ""
        status = f"✅ Generated **{success}** PDB files{err_summary}. Click the download link below."
        return zip_path, status

    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# 5. RNA REFERENCE TAB — GENOME-BACKED VIENNARNA
# ═══════════════════════════════════════════════════════════════

def _extract_and_fold(variant: str, strand: str, genome: Fasta, flank: int) -> dict:
    """Extract ref + mut sequences from genome, fold both with ViennaRNA."""
    chrom, pos, alleles = variant.split(":")
    pos = int(pos)
    ref_allele, mut_allele = alleles.split(">")
    chrom_key = chrom.replace("chr", "")

    left = pos - flank - 1
    right = pos + len(ref_allele) - 1 + flank
    seq = genome[chrom_key][left:right].seq.upper()
    idx = flank

    assert seq[idx:idx + len(ref_allele)] == ref_allele, (
        f"{variant}: reference mismatch"
    )
    mut_seq = seq[:idx] + mut_allele + seq[idx + len(ref_allele):]

    if strand == "-":
        seq = str(BioSeq(seq).reverse_complement())
        mut_seq = str(BioSeq(mut_seq).reverse_complement())

    config = ViennaRNAConfig(temperature=37.0, verbose=0, device="cpu")
    ref_result = run_viennarna(ViennaRNAInput(sequences=[seq]), config)
    mut_result = run_viennarna(ViennaRNAInput(sequences=[mut_seq]), config)

    r = ref_result.results[0]
    m = mut_result.results[0]
    return {
        "variant": variant, "strand": strand,
        "ref_sequence": r.sequence, "ref_structure": r.structure, "ref_mfe": r.mfe,
        "mut_sequence": m.sequence, "mut_structure": m.structure, "mut_mfe": m.mfe,
    }


def handle_ref_upload(file):
    """Parse variant CSV, fold every row, return selector df and initial plots."""
    empty = pd.DataFrame(columns=["#", "Variant", "Strand"])
    if file is None:
        return None, empty, None, None, None, None, "Upload a CSV with columns: **variant**, **strand**."

    if not os.path.exists(REF_GENOME_PATH):
        return None, empty, None, None, None, None, (
            f"❌ Reference genome not found at:\n`{REF_GENOME_PATH}`"
        )

    try:
        df_in = pd.read_csv(file.name)
        df_in.columns = [c.strip().lower() for c in df_in.columns]
        # Support headerless files too
        if "variant" not in df_in.columns:
            df_in = pd.read_csv(file.name, names=["variant", "strand"])

        required = {"variant", "strand"}
        if not required.issubset(set(df_in.columns)):
            return None, empty, None, None, None, None, "❌ CSV must have **variant** and **strand** columns."

        genome = Fasta(REF_GENOME_PATH)
        records = []
        errors = []
        for i, row in df_in.iterrows():
            try:
                records.append(_extract_and_fold(str(row["variant"]).strip(),
                                                  str(row["strand"]).strip(),
                                                  genome, flank=50))
            except Exception as e:
                errors.append(f"Row {i}: {e}")

        if not records:
            return None, empty, None, None, None, None, "❌ No rows processed. Errors:\n" + "\n".join(errors[:5])

        df_out = pd.DataFrame(records)
        selector = df_out[["variant", "strand"]].copy()
        selector.insert(0, "#", range(len(selector)))
        selector.columns = ["#", "Variant", "Strand"]

        r0 = df_out.iloc[0]
        fig3d_ref = generate_3d_plotly(r0["ref_sequence"], r0["ref_structure"], r0["ref_mfe"])
        fig2d_ref = generate_2d_plot(r0["ref_sequence"], r0["ref_structure"])
        fig3d_mut = generate_3d_plotly(r0["mut_sequence"], r0["mut_structure"], r0["mut_mfe"])
        fig2d_mut = generate_2d_plot(r0["mut_sequence"], r0["mut_structure"])
        err_note = (f"\n\n> ⚠️ {len(errors)} rows skipped." if errors else "")
        meta = _build_ref_metadata(r0, 0) + err_note

        return df_out, selector, fig3d_ref, fig2d_ref, fig3d_mut, fig2d_mut, meta

    except Exception as e:
        return None, empty, None, None, None, None, f"❌ Error: {str(e)}"


def handle_ref_row_select(evt: gr.SelectData, df):
    """Row click → update both ref and mut structure panels."""
    if df is None:
        return None, None, None, None, "No data loaded."
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_idx >= len(df):
        return None, None, None, None, f"Row {row_idx} out of range."
    r = df.iloc[row_idx]
    return (
        generate_3d_plotly(r["ref_sequence"], r["ref_structure"], r["ref_mfe"]),
        generate_2d_plot(r["ref_sequence"], r["ref_structure"]),
        generate_3d_plotly(r["mut_sequence"], r["mut_structure"], r["mut_mfe"]),
        generate_2d_plot(r["mut_sequence"], r["mut_structure"]),
        _build_ref_metadata(r, row_idx),
    )


def _build_ref_metadata(row, idx: int) -> str:
    n_ref = len(row["ref_sequence"])
    bp_ref = len(parse_dot_bracket(row["ref_structure"])) // 2
    n_mut = len(row["mut_sequence"])
    bp_mut = len(parse_dot_bracket(row["mut_structure"])) // 2
    return f"""### 📊 Variant Summary — Row {idx}: `{row['variant']}`

| | Reference | Mutant |
|---|---|---|
| **Length** | {n_ref} nt | {n_mut} nt |
| **Base Pairs** | {bp_ref} | {bp_mut} |
| **MFE** | {float(row['ref_mfe']):.4f} kcal/mol | {float(row['mut_mfe']):.4f} kcal/mol |

**Ref structure:** `{row['ref_structure']}`

**Mut structure:** `{row['mut_structure']}`
"""


def _build_metadata(sequence: str, structure: str, mfe: float, idx: int) -> str:
    n = len(sequence)
    bp_map = parse_dot_bracket(structure)
    n_pairs = len(bp_map) // 2
    n_unpaired = n - len(bp_map)

    return f"""### 📊 Structure Summary — Row {idx}

| Property | Value |
|---|---|
| **Nucleotide Length** | $N = {n}$ |
| **Base Pairs** | {n_pairs} pairs ({len(bp_map)} paired residues) |
| **Unpaired Residues** | {n_unpaired} |
| **MFE** | {mfe:.4f} kcal/mol |

**Dot-Bracket Notation:**
```
{structure}
```
"""


# ═══════════════════════════════════════════════════════════════
# 5. GRADIO UI — TABBED LAYOUT (DNA / RNA / Protein)
# ═══════════════════════════════════════════════════════════════

# JavaScript snippet: patch canvas getContext to set willReadFrequently=true
# This keeps Matplotlib canvases in CPU memory so getImageData is fast
# and removes the "Canvas2D: will read frequently" console warning.
CANVAS_PATCH_HEAD = """
<script>
(function() {
    const _origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, attrs) {
        if (type === '2d') {
            attrs = Object.assign({willReadFrequently: true}, attrs || {});
        }
        return _origGetContext.call(this, type, attrs);
    };
})();
</script>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #ffffff !important;
    color: #1f2937 !important;
}
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: #111827 !important;
    font-weight: 600 !important;
}

/* Left pane: natural height, scrollable, right border separator */
.left-pane {
    overflow-y: auto;
    padding-right: 16px;
    border-right: 1px solid #e5e7eb;
}
.left-pane::-webkit-scrollbar { width: 5px; }
.left-pane::-webkit-scrollbar-track { background: #f9fafb; }
.left-pane::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }

/* Right pane: natural height, scrollable vertically only */
.right-pane {
    overflow-y: auto;
    overflow-x: visible;
    padding-left: 16px;
}
.right-pane::-webkit-scrollbar { width: 5px; }
.right-pane::-webkit-scrollbar-track { background: #f9fafb; }
.right-pane::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }

.gr-file, .gr-dropzone {
    border: 1px dashed #e5e7eb !important;
    background: #f9fafb !important;
    border-radius: 6px !important;
}

/* Selector table */
.selector-table {
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    background: white !important;
}
.selector-table table { font-size: 12px !important; }
.selector-table tr:hover {
    background-color: #f3f4f6 !important;
    cursor: pointer;
}

/* Plot containers: enforce minimum height so Plotly doesn't collapse */
.plot-container {
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    background: white !important;
    min-height: 480px !important;
    width: 100% !important;
}
/* Target the inner Plotly div and Matplotlib img directly */
.plot-container > div,
.plot-container .plotly,
.plot-container .js-plotly-plot,
.plot-container img {
    min-height: 480px !important;
    width: 100% !important;
    display: block !important;
}

.meta-card {
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    background: #f9fafb !important;
    padding: 16px !important;
}
button.primary, button.gr-button-primary {
    font-family: 'Inter', sans-serif !important;
    background: #111827 !important;
    color: #ffffff !important;
    border: 1px solid #111827 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
button.primary:hover { background: #374151 !important; }
"""


def build_app():
    with gr.Blocks(
        title="Genomics & Transcriptomics Visualizer",
    ) as demo:

        with gr.Tabs() as global_tabs:

            # ── TAB 1: DNA Analysis ──
            with gr.Tab("DNA Analysis"):
                import alphagenome_UI
                alphagenome_UI.draw_dna_ui()

            # ── TAB 2: RNA Visualizer ──
            with gr.Tab("RNA Structure Visualizer"):
                df_state = gr.State(value=None)

                gr.Markdown(
                    "# 🧬 RNA Secondary Structure Dashboard\n"
                    "Upload a pre-computed `viennarna_results.csv`. "
                    "Click any row to visualize.",
                )

                with gr.Row():
                    # Left Pane (40%)
                    with gr.Column(scale=4, elem_classes=["left-pane"]):
                        gr.Markdown("### 📂 Upload CSV")
                        csv_upload = gr.File(
                            label="Upload CSV (sequence, structure, mfe)",
                            file_types=[".csv"],
                            file_count="single",
                        )

                        gr.Markdown("### 🗂️ Generate PDB Files")
                        gr.Markdown(
                            "Run `generate_pdbs.py` on every row of the loaded CSV "
                            "to produce a folder of `.pdb` 3D structure files, then "
                            "download them as a single zip archive.",
                        )
                        generate_pdbs_btn = gr.Button(
                            "⚙️ Generate PDBs for All Rows",
                            variant="primary",
                            interactive=False,
                        )
                        pdb_status = gr.Markdown(value="")
                        pdb_download = gr.File(
                            label="📦 Download PDB Archive (.zip)",
                            interactive=False,
                            visible=False,
                        )

                        gr.Markdown("### 📋 Sequence Selector")
                        selector_table = gr.Dataframe(
                            value=pd.DataFrame(columns=["Index", "Sequence", "Structure", "MFE"]),
                            headers=["Index", "Sequence", "Structure", "MFE"],
                            interactive=False,
                            wrap=True,
                            elem_classes=["selector-table"],
                        )

                    # Right Pane (60%)
                    with gr.Column(scale=6, elem_classes=["right-pane"]):
                        gr.Markdown("### 🔬 3D Coarse-Grained Viewport")
                        plot_3d = gr.Plot(
                            label="3D Model",
                            elem_classes=["plot-container"],
                        )

                        gr.Markdown("### 🧬 2D Secondary Structure")
                        plot_2d = gr.Plot(
                            label="2D Diagram",
                            elem_classes=["plot-container"],
                        )

                        gr.Markdown("### 📊 Summary Metadata")
                        metadata_panel = gr.Markdown(
                            value="Upload a CSV file to view structural metadata.",
                            elem_classes=["meta-card"],
                        )

                # Callbacks
                def _on_csv_upload(file):
                    df, sel, p3, p2, meta = handle_csv_upload(file)
                    btn_state = gr.update(interactive=df is not None and len(df) > 0)
                    return df, sel, p3, p2, meta, btn_state, "", gr.update(visible=False)

                csv_upload.change(
                    fn=_on_csv_upload,
                    inputs=[csv_upload],
                    outputs=[df_state, selector_table, plot_3d, plot_2d, metadata_panel,
                             generate_pdbs_btn, pdb_status, pdb_download],
                )

                def _run_pdbs(df):
                    zip_path, status = handle_generate_pdbs(df)
                    return (gr.update(interactive=True),
                            gr.update(visible=zip_path is not None, value=zip_path),
                            status)

                generate_pdbs_btn.click(
                    fn=lambda: (gr.update(interactive=False), gr.update(visible=False), "⏳ Generating PDB files…"),
                    inputs=None,
                    outputs=[generate_pdbs_btn, pdb_download, pdb_status],
                ).then(
                    fn=_run_pdbs,
                    inputs=[df_state],
                    outputs=[generate_pdbs_btn, pdb_download, pdb_status],
                )

                selector_table.select(
                    fn=handle_row_select,
                    inputs=[df_state],
                    outputs=[plot_3d, plot_2d, metadata_panel],
                )

            # ── TAB 3: RNA Reference ──
            with gr.Tab("RNA Reference"):
                ref_df_state = gr.State(value=None)

                gr.Markdown(
                    "# 🧬 RNA Reference Structure\n"
                    "Upload a variant CSV (`variant,strand`). Sequences are extracted from the "
                    "GRCh38 reference genome with a 50 nt flank, folded by ViennaRNA, and compared "
                    "side-by-side (reference vs mutant)."
                )

                with gr.Row():
                    # ── Left pane ──
                    with gr.Column(scale=4, elem_classes=["left-pane"]):
                        gr.Markdown("### 📂 Upload Variant CSV")
                        gr.Markdown("`variant` column format: `chr17:43045712:T>C` — `strand` column: `+` or `-`")
                        ref_csv_upload = gr.File(
                            label="Upload CSV (variant, strand)",
                            file_types=[".csv"],
                            file_count="single",
                        )
                        ref_run_status = gr.Markdown(value="")

                        gr.Markdown("### 📋 Variant Selector")
                        ref_selector = gr.Dataframe(
                            value=pd.DataFrame(columns=["#", "Variant", "Strand"]),
                            headers=["#", "Variant", "Strand"],
                            interactive=False,
                            wrap=True,
                            elem_classes=["selector-table"],
                        )

                        gr.Markdown("### 📊 Summary")
                        ref_meta = gr.Markdown(
                            value="Upload a variant CSV to view structural comparison.",
                            elem_classes=["meta-card"],
                        )

                    # ── Right pane ──
                    with gr.Column(scale=6, elem_classes=["right-pane"]):
                        gr.Markdown("### 🔵 Reference Sequence")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**3D Model**")
                                ref_plot_3d = gr.Plot(elem_classes=["plot-container"])
                            with gr.Column():
                                gr.Markdown("**2D Structure**")
                                ref_plot_2d = gr.Plot(elem_classes=["plot-container"])

                        gr.Markdown("### 🔴 Mutant Sequence")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**3D Model**")
                                mut_plot_3d = gr.Plot(elem_classes=["plot-container"])
                            with gr.Column():
                                gr.Markdown("**2D Structure**")
                                mut_plot_2d = gr.Plot(elem_classes=["plot-container"])

                # Callbacks
                def _on_ref_upload(file):
                    df, sel, p3r, p2r, p3m, p2m, meta = handle_ref_upload(file)
                    status = "" if df is not None else meta
                    return df, sel, p3r, p2r, p3m, p2m, meta, status

                ref_csv_upload.change(
                    fn=lambda f: (None, pd.DataFrame(columns=["#","Variant","Strand"]),
                                  None, None, None, None,
                                  "Upload a variant CSV to view structural comparison.",
                                  "⏳ Loading genome & running ViennaRNA…"),
                    inputs=[ref_csv_upload],
                    outputs=[ref_df_state, ref_selector, ref_plot_3d, ref_plot_2d,
                             mut_plot_3d, mut_plot_2d, ref_meta, ref_run_status],
                ).then(
                    fn=_on_ref_upload,
                    inputs=[ref_csv_upload],
                    outputs=[ref_df_state, ref_selector, ref_plot_3d, ref_plot_2d,
                             mut_plot_3d, mut_plot_2d, ref_meta, ref_run_status],
                )

                ref_selector.select(
                    fn=handle_ref_row_select,
                    inputs=[ref_df_state],
                    outputs=[ref_plot_3d, ref_plot_2d, mut_plot_3d, mut_plot_2d, ref_meta],
                )

            # ── TAB 4: Protein Analysis ──
            with gr.Tab("Protein Analysis"):

                gr.Markdown(
                    "# 🧬 ESM2 BRCA1 Missense Variant Scorer\n"
                    "Score missense variants with **ESM2-650M** using "
                    f"**{_esm_device_label()}** hardware acceleration. "
                    "Variants are coloured by log-likelihood ratio (LLR): "
                    "positive → benign-like, negative → pathogenic-like."
                )

                with gr.Row():
                    # ── Left pane ──────────────────────────────────────────
                    with gr.Column(scale=4, elem_classes=["left-pane"]):

                        gr.Markdown("### ⚙️ Configuration")
                        _prot_vep = gr.Textbox(
                            label="VEP TSV path", value=_DEFAULT_VEP, lines=1)
                        _prot_fasta = gr.Textbox(
                            label="Reference FASTA path", value=_DEFAULT_FASTA, lines=1)
                        _prot_model = gr.Textbox(
                            label="HuggingFace model", value=_DEFAULT_MODEL, lines=1)

                        with gr.Row():
                            _prot_window = gr.Slider(
                                label="Max AA window", minimum=64, maximum=1022,
                                value=1022, step=1)
                        with gr.Row():
                            _prot_strict = gr.Checkbox(
                                label="Strict CLIN_SIG labels", value=False)
                            _prot_dedupe = gr.Dropdown(
                                label="Dedupe mode",
                                choices=["id", "mutation"], value="id")

                        _prot_run_btn = gr.Button(
                            "🚀 Run Scoring", variant="primary")

                        gr.Markdown("### 📋 Pipeline Log")
                        _prot_log = gr.Textbox(
                            label="", lines=14, interactive=False,
                            placeholder="Log output will appear here…")
                        _prot_refresh = gr.Button("🔄 Refresh Log")

                        gr.Markdown("### 🔍 Filter Results")
                        with gr.Row():
                            _prot_llr_min = gr.Slider(
                                label="LLR min", minimum=-10, maximum=0,
                                value=-10, step=0.1)
                            _prot_llr_max = gr.Slider(
                                label="LLR max", minimum=0, maximum=10,
                                value=10, step=0.1)
                        _prot_class_filter = gr.CheckboxGroup(
                            label="Clinical class",
                            choices=["benign", "pathogenic"],
                            value=["benign", "pathogenic"])
                        _prot_search = gr.Textbox(
                            label="Search variant / change", placeholder="e.g. R1699Q")
                        _prot_filter_btn = gr.Button("🔍 Apply Filters")

                        _prot_stats = gr.Markdown(
                            value="*Run scoring first, then apply filters.*",
                            elem_classes=["meta-card"])

                        _prot_dl_btn  = gr.Button("⬇️ Download Scores TSV")
                        _prot_dl_file = gr.File(
                            label="Download", visible=False, interactive=False)

                    # ── Right pane ─────────────────────────────────────────
                    with gr.Column(scale=6, elem_classes=["right-pane"]):

                        gr.Markdown("### 🔵 Scatter: Position vs LLR")
                        _prot_scatter = gr.Plot(
                            label="Scatter", elem_classes=["plot-container"])

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### 🎻 Distribution")
                                _prot_violin = gr.Plot(
                                    label="Violin", elem_classes=["plot-container"])
                            with gr.Column():
                                gr.Markdown("### 📈 Summary Stats")
                                _prot_summary = gr.Plot(
                                    label="Summary", elem_classes=["plot-container"])

                        gr.Markdown("### 📊 Top Variants by |LLR|")
                        _prot_top = gr.Plot(
                            label="Top variants", elem_classes=["plot-container"])

                        gr.Markdown("### 🔥 Position Heatmap (50-aa bins)")
                        _prot_heatmap = gr.Plot(
                            label="Heatmap", elem_classes=["plot-container"])

                        gr.Markdown("### 🧫 BRCA1 3D Structure (PDB 1JM7)")
                        gr.Markdown(
                            "_Click **Render 3D** to load. Residues are coloured "
                            "by ESM2 LLR: 🟢 green = benign-like · 🔴 red = pathogenic-like._")
                        _prot_render3d = gr.Button("🎨 Render 3D Structure")
                        _prot_3d = gr.HTML(
                            value="<div style='padding:24px;text-align:center;"
                                  "color:#6b7280;border:1px solid #e5e7eb;"
                                  "border-radius:6px;font-family:sans-serif'>"
                                  "🔬 Click <b>Render 3D Structure</b> to load the BRCA1 viewer.</div>")

                        gr.Markdown("### 📋 Variant Table")
                        _prot_table = gr.Dataframe(
                            interactive=False, wrap=True,
                            elem_classes=["selector-table"])

                # ── Callbacks ───────────────────────────────────────────────
                def _prot_start(vep, fasta, model, win, strict, dedupe):
                    if _ESM_STATE["running"]:
                        return "⚠️ Already running — please wait."
                    threading.Thread(
                        target=_esm_run_scoring,
                        args=(vep, fasta, model, int(win), strict, dedupe),
                        daemon=True,
                    ).start()
                    return "🚀 Scoring started — click Refresh Log to track progress."

                _prot_run_btn.click(
                    fn=_prot_start,
                    inputs=[_prot_vep, _prot_fasta, _prot_model,
                            _prot_window, _prot_strict, _prot_dedupe],
                    outputs=[_prot_log],
                )

                _prot_refresh.click(
                    fn=lambda: "\n".join(_ESM_STATE["log"][-60:]),
                    inputs=[], outputs=[_prot_log],
                )

                def _prot_apply_filters(llr_min, llr_max, cls_filter, query):
                    df = _ESM_STATE["scores"]
                    empty_fig = go.Figure()
                    if df is None or df.empty:
                        return (empty_fig, empty_fig, empty_fig, empty_fig,
                                empty_fig, pd.DataFrame(),
                                "*No data yet — run scoring first.*")

                    mask = (
                        (df["log_likelihood_ratio"] >= llr_min)
                        & (df["log_likelihood_ratio"] <= llr_max)
                    )
                    if cls_filter:
                        mask &= df["clinical_class"].isin(cls_filter)
                    if query.strip():
                        q = query.strip().lower()
                        mask &= (
                            df["variant_id"].str.lower().str.contains(q, na=False)
                            | df["protein_change"].str.lower().str.contains(q, na=False)
                        )
                    filt = df[mask].copy()

                    if filt.empty:
                        return (empty_fig, empty_fig, empty_fig, empty_fig,
                                empty_fig, pd.DataFrame(),
                                "_No variants match the current filters._")

                    n_b = int((filt["clinical_class"] == "benign").sum())
                    n_p = int((filt["clinical_class"] == "pathogenic").sum())
                    mb  = filt[filt["clinical_class"] == "benign"]["log_likelihood_ratio"].mean()
                    mp  = filt[filt["clinical_class"] == "pathogenic"]["log_likelihood_ratio"].mean()
                    stats_md = (
                        f"**Filtered:** {len(filt)} variants &nbsp;|&nbsp; "
                        f"Benign: {n_b} (mean LLR {mb:.3f}) &nbsp;|&nbsp; "
                        f"Pathogenic: {n_p} (mean LLR {mp:.3f})"
                    )

                    cols = [c for c in [
                        "variant_id", "protein_change", "clinical_class",
                        "log_likelihood_ratio", "fold_change",
                        "ref_aa", "alt_aa", "protein_position",
                    ] if c in filt.columns]

                    return (
                        _esm_fig_scatter(filt),
                        _esm_fig_violin(filt),
                        _esm_fig_summary(filt),
                        _esm_fig_top(filt, n=30),
                        _esm_fig_heatmap(filt),
                        filt[cols].sort_values("log_likelihood_ratio"),
                        stats_md,
                    )

                _prot_filter_btn.click(
                    fn=_prot_apply_filters,
                    inputs=[_prot_llr_min, _prot_llr_max,
                            _prot_class_filter, _prot_search],
                    outputs=[_prot_scatter, _prot_violin, _prot_summary,
                             _prot_top, _prot_heatmap,
                             _prot_table, _prot_stats],
                )

                _prot_render3d.click(
                    fn=lambda: _esm_3d_viewer(_ESM_STATE.get("scores")),
                    inputs=[], outputs=[_prot_3d],
                )

                def _prot_download():
                    df = _ESM_STATE.get("scores")
                    if df is None or df.empty:
                        return None
                    out = os.path.join(os.path.dirname(__file__), "esm2_scores_export.tsv")
                    df.to_csv(out, sep="\t", index=False)
                    return out

                _prot_dl_btn.click(
                    fn=_prot_download,
                    inputs=[], outputs=[_prot_dl_file],
                )


    return demo


# ═══════════════════════════════════════════════════════════════
# 6. LAUNCH
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_port=5173,
        theme=gr.themes.Default(
            primary_hue="gray",
            secondary_hue="gray",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
        head=CANVAS_PATCH_HEAD,
    )
