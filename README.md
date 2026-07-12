# DOGMA

DOGMA is an experimental bioinformatics workspace for following variant effects
across DNA, RNA, and protein analysis.

## Main components

- `dogma_gradio/` — the structured Gradio application and DOGMA pipeline.
- `alphagenome_UI.py` and `alphagenome_UI2.py` — AlphaGenome interface prototypes.
- `Notebook_format.ipynb` — combined ESM-2, ViennaRNA, AlphaGenome, and Ensembl experiments.
- `alphagenome_colab.ipynb` and `alphagenome_vep_notebook.ipynb` — AlphaGenome experiments.
- `esm_notebook.ipynb` and `vienna_rna.ipynb` — protein and RNA model experiments.
- `NOTES/` — development and pipeline notes.

Raw datasets, generated result files, local credentials, and caches are excluded
from Git. See `dogma_gradio/README.md` for application setup and usage.

## Credentials

Set the AlphaGenome key locally; never commit it:

```bash
export ALPHAGENOME_API_KEY="your-key"
```
