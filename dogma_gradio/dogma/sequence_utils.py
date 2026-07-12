from __future__ import annotations

import re

from Bio.Seq import Seq

from .models import VariantInput


_DNA_RE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)


def validate_variant(variant: VariantInput, *, substitutions_only: bool = True) -> VariantInput:
    """Validate and normalize a variant used by the prototype pipeline."""
    variant = variant.normalized()

    if variant.position < 1:
        raise ValueError("Variant position must be a positive 1-based coordinate.")

    if not variant.reference_bases or not variant.alternate_bases:
        raise ValueError("REF and ALT alleles cannot be empty.")

    if not _DNA_RE.fullmatch(variant.reference_bases):
        raise ValueError("REF may contain only A, C, G, T, or N.")

    if not _DNA_RE.fullmatch(variant.alternate_bases):
        raise ValueError("ALT may contain only A, C, G, T, or N.")

    if substitutions_only and len(variant.reference_bases) != len(variant.alternate_bases):
        raise ValueError(
            "This first DOGMA UI supports substitutions/MNVs only for the full "
            "DNA→RNA→protein path. REF and ALT must have equal lengths."
        )

    return variant


def normalize_chromosome_for_ensembl(chromosome: str) -> str:
    chromosome = str(chromosome).strip()
    if chromosome.lower().startswith("chr"):
        chromosome = chromosome[3:]
    if chromosome.upper() == "M":
        chromosome = "MT"
    return chromosome


def strip_version(identifier: str | None) -> str | None:
    if identifier is None:
        return None
    return str(identifier).split(".", 1)[0]


def reverse_complement(sequence: str) -> str:
    return str(Seq(str(sequence).upper()).reverse_complement())


def dna_to_rna(sequence: str) -> str:
    return str(sequence).upper().replace("T", "U")


def orient_dna_to_transcript(sequence: str, strand: str | int) -> str:
    """Return DNA in transcript 5'→3' orientation."""
    is_negative = strand in (-1, "-", "-1")
    return reverse_complement(sequence) if is_negative else str(sequence).upper()


def translate_cds(cds_sequence: str) -> str:
    """Translate a CDS, removing a normal terminal stop and truncating at a premature stop."""
    cds_sequence = str(cds_sequence).upper()
    usable_length = (len(cds_sequence) // 3) * 3
    protein = str(Seq(cds_sequence[:usable_length]).translate(to_stop=False))

    if protein.endswith("*"):
        protein = protein[:-1]
    if "*" in protein:
        protein = protein[: protein.index("*") + 1]
    return protein


def parse_csv_text(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]
