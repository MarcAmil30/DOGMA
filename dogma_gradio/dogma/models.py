from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantInput:
    """A genomic variant using VCF/AlphaGenome-style 1-based coordinates."""

    chromosome: str
    position: int
    reference_bases: str
    alternate_bases: str

    def normalized(self) -> "VariantInput":
        chromosome = str(self.chromosome).strip()
        if not chromosome.lower().startswith("chr"):
            chromosome = f"chr{chromosome}"

        return VariantInput(
            chromosome=chromosome,
            position=int(self.position),
            reference_bases=str(self.reference_bases).strip().upper(),
            alternate_bases=str(self.alternate_bases).strip().upper(),
        )

    @property
    def label(self) -> str:
        return (
            f"{self.chromosome}:{self.position}:"
            f"{self.reference_bases}>{self.alternate_bases}"
        )
