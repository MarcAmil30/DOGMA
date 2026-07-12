from dogma.models import VariantInput
from dogma.sequence_utils import (
    dna_to_rna,
    orient_dna_to_transcript,
    validate_variant,
)


def test_variant_normalization():
    variant = validate_variant(VariantInput("22", 10, "a", "c"))
    assert variant.chromosome == "chr22"
    assert variant.reference_bases == "A"
    assert variant.alternate_bases == "C"


def test_negative_strand_rna_orientation():
    # Forward genomic DNA 5'-AGTC-3' becomes transcript-oriented 5'-GACT-3'.
    oriented = orient_dna_to_transcript("AGTC", "-")
    assert oriented == "GACT"
    assert dna_to_rna(oriented) == "GACU"
