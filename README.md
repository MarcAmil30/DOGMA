# AI x DNA Hackathon: Genomics, Transcriptomics & Proteomics Visualizer

Welcome to the **Genomics, Transcriptomics & Proteomics Visualizer** workspace. This repository contains a multi-modal bioinformatics dashboard built with Gradio and a suite of supporting physical simulation and deep learning backend scripts. It was designed to analyze and visualize the functional effects of genetic variants across DNA, RNA, and protein domains.

---

## 📂 Repository Structure

The workspace is organized as follows:

*   **[app.py](file:///Users/phamngoctu/crick%20hackathon/app.py)** — Main Gradio dashboard integrating DNA (AlphaGenome DOGMA), RNA (ViennaRNA & force-directed layout), and Protein (ESM2 missense variant scoring) visualisers.
*   **[alphagenome_UI.py](file:///Users/phamngoctu/crick%20hackathon/alphagenome_UI.py)** — Gradio UI components and functions wrapping Google DeepMind's AlphaGenome model client to perform DOGMA omics variant predictions.
*   **[score_esm2_missense_likelihoods.py](file:///Users/phamngoctu/crick%20hackathon/score_esm2_missense_likelihoods.py)** — Backend script for scoring missense variant pathogenicity using the ESM2 masked protein language model.
*   **[generate_pdbs.py](file:///Users/phamngoctu/crick%20hackathon/generate_pdbs.py)** — Physical spring-relaxation model to fold RNA structures from dot-bracket notation into standard 3D PDB coordinates.
*   **[rna_structure.py](file:///Users/phamngoctu/crick%20hackathon/rna_structure.py)** — CLI utility script to extract sequence windows around genomic variants, apply mutations, and fold them using ViennaRNA.
*   **[requirements.txt](file:///Users/phamngoctu/crick%20hackathon/requirements.txt)** — Python packaging requirements (PyTorch, Transformers, Biopython, Gradio, Plotly, etc.).
*   **[BRCA1_reference.fasta](file:///Users/phamngoctu/crick%20hackathon/BRCA1_reference.fasta)** — Reference protein sequence file for BRCA1.
*   **[45vra2OTHvR2t6dy.Consequence_is_missense_variant.txt](file:///Users/phamngoctu/crick%20hackathon/45vra2OTHvR2t6dy.Consequence_is_missense_variant.txt)** — Tab-separated VEP (Variant Effect Predictor) export containing annotated missense variants.
*   **[fake_input.csv](file:///Users/phamngoctu/crick%20hackathon/fake_input.csv)** — Sample input CSV for testing the ViennaRNA genomic folding pipelines.

---

## 🛠️ Installation & Setup

Ensure you have **Python 3.9+** installed. Set up a virtual environment and install the required dependencies:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Dependencies
The dependencies managed in **[requirements.txt](file:///Users/phamngoctu/crick%20hackathon/requirements.txt)** are:
*   **Core ML:** `torch`, `transformers`
*   **Data & Science:** `pandas`, `numpy`, `biopython`, `scipy`
*   **Visualization:** `matplotlib`, `plotly`
*   **Web App & UI:** `gradio>=4.0`
*   **CLI Utilities:** `tqdm`, `pyfaidx`

### Local Hardware Acceleration
Pathogenicity scoring with ESM2 automatically detects and uses hardware acceleration for faster inference:
1.  **NVIDIA GPUs (CUDA)**
2.  **Apple Silicon GPUs (Metal Performance Shaders / MPS)**
3.  **CPU** (Fallback)

---

## 📊 Dashboard Modules (Gradio UI)

Run the Gradio dashboard using:
```bash
python app.py
```
This launches a browser-based user interface with three primary analytical sections:

### 1. 🧬 DNA Tab (AlphaGenome DOGMA Scoring)
*   **Model:** Powered by Google DeepMind's AlphaGenome.
*   **Description:** Scores human variants (hg38 coordinates) to predict their downstream epigenetic and transcription impacts.
*   **Outputs Visualized:**
    *   RNA-seq transcript levels.
    *   Splice sites and splice site usage.
    *   CAGE (Cap Analysis Gene Expression) transcription start sites.
    *   Chromatin accessibility profiles (DNase, ATAC).
*   **Interface:** Features interactive options for genomic interval resizing, tissue ontology filters (e.g., EFO, GTEx), and generates side-by-side reference vs. alternate overlaid track plots.

### 2. 🌀 RNA Tab (Secondary Structure & Folding)
*   **Visualizations:**
    *   **2D Structure Plot:** Renders a layout of the RNA secondary structure colored by structural element types (`stem`, `hairpin`, `interior loop / bulge`, `multiloop junction`, and `exterior dangling`).
    *   **3D Coarse-Grained Plot:** Renders an interactive Plotly 3D scatter plot of the folded RNA backbone.
*   **PDB Exporter:** Automatically folds and writes structural `.pdb` files for bulk datasets using the force-directed physics engine.
*   **Genome-Backed Folding:**
    *   Extracts sequence context around a variant from a primary genome assembly file (`Homo_sapiens.GRCh38.dna.primary_assembly.fa`).
    *   Folds both the reference and mutated variants using ViennaRNA.
    *   Displays structural element differences and Minimum Free Energy (MFE) delta calculations.

### 3. 🔬 Protein Tab (ESM2 Missense Pathogenicity)
*   **Model:** Powered by `facebook/esm2_t33_650M_UR50D`.
*   **Description:** Evaluates the functional impact of missense mutations by performing masked language model scoring. For each variant, ESM2 yields:
    $$\text{Log-Likelihood Ratio (LLR)} = \log P(\text{mutant}) - \log P(\text{wild-type})$$
*   **Outputs Visualized:**
    *   **LLR along coordinates:** Scatter plot showing LLR values across protein residue positions.
    *   **Pathogenicity Distribution:** Violin plots contrasting LLRs between benign and pathogenic variants.
    *   **Top Variants:** Bar charts of the most damaging mutations ranked by $|LLR|$.
    *   **3D Protein Viewer:** Embedded 3Dmol.js viewer rendering the protein structure (PDB ID: 1JM7 for BRCA1), where residues are colored by the ESM2-predicted LLR score (Red: pathogenic/damaging, Gray: neutral, Teal/Green: benign).

---

## ⚙️ CLI Scripts & Pipelines

### ESM2 Missense Variant Pathogenicity Evaluator
To execute ESM2 variant scoring directly from the terminal, use **[score_esm2_missense_likelihoods.py](file:///Users/phamngoctu/crick%20hackathon/score_esm2_missense_likelihoods.py)**:

```bash
python score_esm2_missense_likelihoods.py \
  --vep 45vra2OTHvR2t6dy.Consequence_is_missense_variant.txt \
  --fasta BRCA1_reference.fasta \
  --out_prefix brca1_scoring \
  --model facebook/esm2_t33_650M_UR50D \
  --device mps
```

*   **Key Arguments:**
    *   `--vep`: Path to VEP-annotated variants file.
    *   `--fasta`: Path to canonical reference protein FASTA.
    *   `--strict_labels`: Enable strict filter for ClinVar labels (omitting conflicting/VUS classifications).
    *   `--max_aa_window`: Maximum size of the amino acid window centered around the variant.
*   **Output Files Generated:**
    *   `*.scores.tsv`: Full set of calculated LLR, probability values, and transcript selections.
    *   `*.summary.tsv`: Statistical summaries (mean, median, count) segmented by benign vs. pathogenic classifications.
    *   `*.skipped.tsv`: Record of variant rows skipped due to unsupported consequences, invalid coordinates, or sequence mismatching.

### Physical RNA 3D PDB Generator
The script **[generate_pdbs.py](file:///Users/phamngoctu/crick%20hackathon/generate_pdbs.py)** uses a force-directed spring relaxation algorithm to simulate RNA physics:
1.  **Backbone Continuity:** Sequential residues are forced ~4.0 Å apart.
2.  **Base Pairing:** Paired residues are pulled to ~2.8 Å.
3.  **Steric Hindrance:** Non-bonded residues repel one another at distances < 6.5 Å.
4.  **Origin Centering:** A global gravity force prevents structural drift.

Run the generator in bulk from a ViennaRNA results table:
```bash
python generate_pdbs.py \
  --input viennarna_results.csv \
  --output_dir dogma_outputs \
  --iterations 150
```

---

## 🤝 Authors & Credits
*   **Antigravity** — Lead Bioinformatics Developer.
*   *Built for the Crick AI x DNA Hackathon.*
