# DOGMA

DOGMA is an experimental bioinformatics project for studying how one human
genomic variant may affect DNA-related signals, local RNA folding, and protein
sequence likelihood.

The repository has two main parts:

- the notebooks in the project root are exploratory examples and development
  work;
- `dogma_gradio/` is the structured application that connects the analyses in
  one Gradio user interface.

## Repository structure

```text
DOGMA/
├── README.md
├── Notebook_format.ipynb
├── alphagenome_vep_notebook.ipynb
├── esm_notebook.ipynb
├── data/
│   ├── ClinVar_benign_data.txt
│   └── ClinVar_pathogenic_data.txt
├── NOTES/
│   └── 11-07-2026-Dogma_pipeline.md
└── dogma_gradio/
    ├── app.py
    ├── README.md
    ├── requirements.txt
    ├── dogma/
    │   ├── __init__.py
    │   ├── models.py
    │   ├── sequence_utils.py
    │   ├── alphagenome_service.py
    │   ├── ensembl_service.py
    │   ├── vienna_service.py
    │   ├── esm_service.py
    │   └── pipeline.py
    ├── tests/
    │   └── test_sequence_utils.py
    └── outputs/
        └── .gitkeep
```

Generated caches and local environments such as `__pycache__/`,
`.pytest_cache/`, and `.venv/` are not part of the project structure.

## What the top-level files do

### Notebooks

- `Notebook_format.ipynb` is the combined development notebook. It experiments
  with ESM2 protein scoring, ViennaRNA folding, AlphaGenome scoring, and Ensembl
  variant/transcript mapping. Much of the reusable logic was later separated
  into the modules under `dogma_gradio/dogma/`.
- `alphagenome_vep_notebook.ipynb` is an AlphaGenome example. It defines a
  variant, scores its predicted effects across selected biological tracks,
  exports scores, and plots reference-versus-alternate predictions around the
  variant.
- `esm_notebook.ipynb` demonstrates the `proto_tools` ESM2 functions. It shows
  how to create protein embeddings, propose mutations, score protein sequences,
  calculate gradients, and export results. It is a learning/example notebook;
  the Gradio pipeline uses only ESM2 sequence scoring.

### Data and notes

- `data/ClinVar_benign_data.txt` and
  `data/ClinVar_pathogenic_data.txt` are tab-separated Ensembl VEP annotation
  tables for benign and pathogenic/likely pathogenic ClinVar variants. They are
  not VCF files. The `.gitignore` currently excludes `.txt` data, so these files
  remain local unless that rule is changed deliberately.
- `NOTES/11-07-2026-Dogma_pipeline.md` records the original design request and
  reasoning behind joining the three modalities into a UI.

## How the Gradio application is organised

`dogma_gradio/app.py` contains the user interface. It collects the API key,
variant, AlphaGenome tracks, ontology terms, RNA window size, optional gene,
and ESM2 settings. It passes those values to the pipeline and displays the
returned tables. It does not perform biological analysis itself.

The `dogma_gradio/dogma/` package contains the analysis code:

- `models.py` defines `VariantInput`, the shared representation of chromosome,
  1-based position, reference allele, and alternate allele. It also normalises
  inputs such as `22` to `chr22` and lowercase bases to uppercase.
- `sequence_utils.py` validates alleles and coordinates and provides shared DNA,
  RNA, strand, translation, and identifier helper functions.
- `alphagenome_service.py` creates an AlphaGenome variant, runs the selected
  recommended scorers, converts their output to a table, filters it by track and
  ontology, and ranks the strongest results. It can select the strongest linked
  protein-coding gene for the later RNA and protein steps.
- `ensembl_service.py` talks to the Ensembl REST API. It verifies the GRCh38
  reference allele with VEP, finds affected genes and transcripts, fetches a
  genomic sequence window for RNA folding, retrieves translated isoforms, and
  reconstructs alternate proteins when the variant maps directly and safely to
  a transcript CDS. Identical protein sequences are grouped into isoform IDs.
- `vienna_service.py` places the fetched reference and alternate DNA windows in
  the gene's 5'-to-3' direction, converts them to RNA, folds both with
  ViennaRNA, and reports minimum free energy (MFE) and `ALT MFE - REF MFE`.
- `esm_service.py` masks each changed amino-acid position in the reference
  protein and asks ESM2 for the reference and alternate residue probabilities
  in one pass. It reports the alternate-minus-reference positional
  log-probability and skips changes it cannot represent safely.
- `pipeline.py` is the coordinator. It runs Ensembl VEP, AlphaGenome, gene and
  isoform selection, ViennaRNA, and ESM2 in order. It preserves partial results
  if an individual stage fails, writes the output tables and metadata, creates a
  ZIP archive, and returns everything to the UI.
- `__init__.py` marks the directory as an importable Python package.

Supporting application files:

- `requirements.txt` lists the public Python dependencies. The project-specific
  `proto_tools` package must be installed separately because it supplies the
  ViennaRNA and ESM2 wrappers used here.
- `tests/test_sequence_utils.py` checks variant normalisation and negative-strand
  RNA orientation.
- `outputs/` receives one timestamped result directory and one ZIP file for each
  run. `.gitkeep` keeps the otherwise empty directory in Git.
- `dogma_gradio/README.md` contains detailed installation, test input, output,
  and current-limitation notes for the application.

## Pipeline flow

```text
Variant entered in Gradio
        |
        v
Validate and normalise the input
        |
        v
Ensembl VEP verifies REF and identifies affected transcripts
        |
        +-------------------+--------------------+
        |                   |                    |
        v                   v                    v
AlphaGenome scores    Ensembl RNA window   Ensembl protein isoforms
DNA-related tracks          |                    |
        |                   v                    v
        |             ViennaRNA folds      ESM2 scores REF/ALT
        |              REF and ALT          proteins when resolved
        |                   |                    |
        +-------------------+--------------------+
                            |
                            v
              Tables, metadata, and ZIP output
```

In simple terms, AlphaGenome estimates changes in genomic activity, ViennaRNA
compares local RNA structures, and ESM2 compares how plausible the reference
and alternate protein sequences look to a protein language model.

## Running the application

```bash
cd dogma_gradio
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Install the same `proto_tools` package used by the notebooks, then provide an
AlphaGenome API key without committing it:

```bash
export ALPHAGENOME_API_KEY="your-key"
python app.py
```

Open `http://127.0.0.1:7860`. See `dogma_gradio/README.md` for a ready-to-use
test variant and more detail.

## Current scope

- Human GRCh38 variants only.
- The complete path supports substitutions and equal-length multi-nucleotide
  variants, not insertions or deletions.
- ViennaRNA folds a local genomic/pre-mRNA window, not a fully spliced mature
  transcript.
- Alternate proteins are created only when the CDS change can be mapped safely.
- AlphaGenome and Ensembl require internet access; ESM2 and ViennaRNA run
  through the locally installed `proto_tools` package.
- Model scores are research signals, not clinical diagnoses or pathogenicity
  probabilities.
