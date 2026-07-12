from __future__ import annotations

import pandas as pd

from .ensembl_service import GenomicWindow
from .sequence_utils import dna_to_rna, orient_dna_to_transcript


def run_vienna_on_genomic_window(
    *,
    window: GenomicWindow,
    gene_name: str,
    gene_strand: str,
    temperature: float = 37.0,
) -> pd.DataFrame:
    """
    Fold strand-oriented REF and ALT RNA windows.

    This is a local genomic/pre-mRNA proxy. It is not a fully spliced mature-mRNA
    model when the window contains intronic sequence.
    """
    try:
        from proto_tools import ViennaRNAConfig, ViennaRNAInput, run_viennarna
    except ImportError as exc:
        raise RuntimeError(
            "proto_tools is not installed in this Python environment. Install the "
            "same proto_tools package/environment used by your working notebooks."
        ) from exc

    ref_oriented_dna = orient_dna_to_transcript(window.reference_dna, gene_strand)
    alt_oriented_dna = orient_dna_to_transcript(window.alternate_dna, gene_strand)
    ref_rna = dna_to_rna(ref_oriented_dna)
    alt_rna = dna_to_rna(alt_oriented_dna)

    inputs = ViennaRNAInput(sequences=[ref_rna, alt_rna])
    config = ViennaRNAConfig(temperature=float(temperature))
    output = run_viennarna(inputs, config)

    results = list(getattr(output, "results", []))
    if len(results) != 2:
        raise RuntimeError(
            f"ViennaRNA returned {len(results)} result(s); expected REF and ALT."
        )

    rows = []
    labels = ["REF", "ALT"]
    for label, result in zip(labels, results, strict=True):
        sequence = str(getattr(result, "sequence", ref_rna if label == "REF" else alt_rna))
        structure = str(getattr(result, "structure", ""))
        mfe = float(getattr(result, "mfe"))
        rows.append(
            {
                "gene_name": gene_name,
                "gene_strand": gene_strand,
                "sequence_context": "strand-aware genomic/pre-mRNA window",
                "region": f"{window.chromosome}:{window.start}-{window.end}",
                "allele": label,
                "rna_length": len(sequence),
                "rna_sequence_5_to_3": sequence,
                "dot_bracket_structure": structure,
                "mfe_kcal_per_mol": mfe,
            }
        )

    df = pd.DataFrame(rows)
    ref_mfe = float(df.loc[df["allele"] == "REF", "mfe_kcal_per_mol"].iloc[0])
    df["delta_mfe_vs_ref"] = df["mfe_kcal_per_mol"] - ref_mfe
    return df
