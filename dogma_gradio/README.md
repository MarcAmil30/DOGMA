# DOGMA Gradio prototype

A modular **DNA → RNA → protein** variant pipeline for a single human GRCh38
substitution/MNV.

## File structure

```text
dogma_gradio/
├── app.py                         # Gradio UI only
├── requirements.txt
├── README.md
├── dogma/
│   ├── models.py                  # Shared variant data model
│   ├── sequence_utils.py          # Validation, strand orientation, translation
│   ├── alphagenome_service.py     # Selected AlphaGenome scorers + filtering
│   ├── ensembl_service.py         # VEP, genome sequence, transcripts, isoforms
│   ├── vienna_service.py          # REF/ALT RNA folding
│   ├── esm_service.py             # Reference/alternate protein scoring
│   └── pipeline.py                # Connects all three modalities
├── tests/
│   └── test_sequence_utils.py
└── outputs/                       # One timestamped folder and ZIP per run
```

The UI never contains analysis logic. Each model/API has one service file, and
`pipeline.py` is the only place that connects them.

## Installation

```bash
cd dogma_gradio
python -m venv .venv
source .venv/bin/activate           # macOS/Linux
python -m pip install -U pip
pip install -r requirements.txt
```

Install your existing project-specific `proto_tools` package in this same
environment. Verify it with:

```bash
python -c "from proto_tools import run_viennarna, run_esm2_score; print('proto_tools OK')"
```

Run the app:

```bash
python app.py
```

Open `http://127.0.0.1:7860`.

## Test case

Use:

```text
Chromosome: chr22
Position:   36201698
REF:        A
ALT:        C
Ontology:   UBERON:0002046
Tracks:     RNA_SEQ, ATAC
Window:     50 bp on each side
```

The gene can be left blank for automatic selection or explicitly set to
`APOL4`.

## Biological flow

### 1. AlphaGenome

The app constructs one `genome.Variant`, resizes its reference interval to a
supported AlphaGenome length, runs only the selected recommended variant
scorers, converts the result to a tidy table, and then filters the table by the
requested ontology CURIE(s).

Important: `score_variant` produces scores for all tracks inside each selected
scorer. In this prototype, ontology filtering is applied to the tidy score table
rather than reducing inference to one ontology before scoring.

### 2. ViennaRNA

The app fetches a forward-strand GRCh38 window from Ensembl and validates the
supplied REF allele. It creates the ALT genomic sequence, then uses the selected
gene's strand:

- positive-strand gene: keep the sequence as fetched;
- negative-strand gene: reverse-complement the entire window;
- convert `T` to `U`;
- fold REF and ALT and report MFE plus `ALT MFE - REF MFE`.

This is a **strand-aware genomic/pre-mRNA local folding proxy**. If the variant
is intronic, the window contains intronic sequence. A mature-mRNA analysis would
instead need a transcript-specific spliced cDNA window for each isoform.

### 3. Protein and ESM2

The app does not blindly mutate every protein attached to an AlphaGenome gene
name. Ensembl VEP maps the variant to every translated transcript. The app:

1. retrieves every translated Ensembl transcript/protein;
2. groups identical reference protein sequences into isoform IDs;
3. reconstructs an alternate protein only when VEP provides a direct CDS
   coordinate and the transcript CDS reference allele matches;
4. leaves intronic, regulatory, and splice-only outcomes unresolved rather than
   inventing a protein sequence;
5. scores reference proteins and resolvable alternate proteins with ESM2;
6. reports `ALT - REF` average log-likelihood.

A positive ESM delta means the alternate sequence is more likely under ESM2. It
is not a pathogenicity probability and does not replace experimental or clinical
interpretation.

## API-key handling

The AlphaGenome key is entered in a password field (or read from the
`ALPHAGENOME_API_KEY` environment variable) and is passed directly to the
client for that run. It is not written to CSV, JSON, or ZIP output. For a shared
or deployed app, use a server-side secret/environment variable instead of asking
end users to paste a key.

## Outputs

Every run creates:

- `dogma_summary.csv`
- `alphagenome_scores.csv`
- `viennarna_scores.csv`
- `ensembl_isoforms.csv`
- `protein_sequences.csv`
- `esm2_scores.csv`
- `run_metadata.json`
- one ZIP containing all files

## Current limitations

- Human GRCh38 only.
- Equal-length substitutions/MNVs only for the complete pipeline.
- ViennaRNA currently uses a genomic/pre-mRNA window, not spliced cDNA.
- ESM2 scores only proteins no longer than the UI maximum.
- Model/API calls require internet access except local ESM2/ViennaRNA execution.
