from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from .models import VariantInput
from .sequence_utils import (
    normalize_chromosome_for_ensembl,
    reverse_complement,
    strip_version,
    translate_cds,
)


ENSEMBL_SERVER = "https://rest.ensembl.org"


@dataclass
class GenomicWindow:
    chromosome: str
    start: int
    end: int
    variant_index: int
    reference_dna: str
    alternate_dna: str


class EnsemblClient:
    """Small Ensembl REST client used by the DOGMA prototype."""

    def __init__(self, species: str = "homo_sapiens") -> None:
        self.species = species
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "DOGMA-Gradio/0.1"})

    def _request(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str = "application/json",
        retries: int = 5,
    ) -> requests.Response:
        url = f"{ENSEMBL_SERVER}{endpoint}"
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers={"Content-Type": accept, "Accept": accept},
                    timeout=60,
                )

                if response.status_code == 429:
                    time.sleep(float(response.headers.get("Retry-After", 1)))
                    continue
                if 500 <= response.status_code < 600:
                    time.sleep(2**attempt)
                    continue

                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Ensembl request failed: {url}") from last_error

    def get_json(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        return self._request(endpoint, params=params).json()

    def get_text(self, endpoint: str, params: dict[str, Any] | None = None) -> str:
        return self._request(endpoint, params=params, accept="text/plain").text.strip()

    def lookup_gene(self, symbol: str, *, expand: bool = True) -> dict[str, Any]:
        symbol = str(symbol).strip()
        if not symbol:
            raise ValueError("Gene symbol is empty.")
        return self.get_json(
            f"/lookup/symbol/{self.species}/{quote(symbol)}",
            params={"expand": 1 if expand else 0},
        )

    def get_sequence(self, identifier: str, sequence_type: str) -> str:
        return self.get_text(
            f"/sequence/id/{identifier}",
            params={"type": sequence_type},
        ).upper()

    def annotate_variant(self, variant: VariantInput) -> dict[str, Any]:
        """Annotate an equal-length substitution with Ensembl VEP."""
        chromosome = normalize_chromosome_for_ensembl(variant.chromosome)
        end = variant.position + len(variant.reference_bases) - 1
        region = f"{chromosome}:{variant.position}-{end}"

        results = self.get_json(
            f"/vep/{self.species}/region/{region}/{variant.alternate_bases}",
            params={
                "symbol": 1,
                "canonical": 1,
                "mane": 1,
                "appris": 1,
                "ccds": 1,
                "protein": 1,
                "hgvs": 1,
                "numbers": 1,
            },
        )
        if not results:
            raise ValueError("Ensembl VEP returned no result for this variant.")

        result = results[0]
        allele_string = str(result.get("allele_string", ""))
        if "/" in allele_string:
            genome_ref = allele_string.split("/", 1)[0].upper()
            if genome_ref != variant.reference_bases:
                raise ValueError(
                    "GRCh38 reference mismatch: "
                    f"supplied REF={variant.reference_bases}, Ensembl REF={genome_ref}."
                )
        return result

    def candidate_gene_symbols(self, vep_result: dict[str, Any]) -> list[str]:
        """Return unique protein-coding gene symbols implicated by VEP."""
        symbols: list[str] = []
        seen: set[str] = set()
        for item in vep_result.get("transcript_consequences", []):
            symbol = item.get("gene_symbol")
            biotype = item.get("biotype")
            if symbol and biotype == "protein_coding" and symbol not in seen:
                symbols.append(str(symbol))
                seen.add(str(symbol))
        return symbols

    def extract_genomic_window(self, variant: VariantInput, flank_size: int) -> GenomicWindow:
        """Fetch a GRCh38 forward-strand window and construct the ALT sequence."""
        flank_size = int(flank_size)
        if flank_size < 0:
            raise ValueError("ViennaRNA flank size cannot be negative.")

        chromosome = normalize_chromosome_for_ensembl(variant.chromosome)
        start = variant.position - flank_size
        end = variant.position + len(variant.reference_bases) - 1 + flank_size
        if start < 1:
            raise ValueError("Requested sequence window starts before coordinate 1.")

        region = f"{chromosome}:{start}..{end}:1"
        reference_dna = self.get_text(
            f"/sequence/region/{self.species}/{region}",
            params={"coord_system_version": "GRCh38"},
        ).upper()

        expected_length = 2 * flank_size + len(variant.reference_bases)
        if len(reference_dna) != expected_length:
            raise ValueError(
                f"Unexpected Ensembl sequence length {len(reference_dna)}; "
                f"expected {expected_length}."
            )

        variant_index = flank_size
        observed_ref = reference_dna[
            variant_index : variant_index + len(variant.reference_bases)
        ]
        if observed_ref != variant.reference_bases:
            raise ValueError(
                f"Reference mismatch at {variant.chromosome}:{variant.position}: "
                f"expected {variant.reference_bases}, fetched {observed_ref}."
            )

        alternate_dna = (
            reference_dna[:variant_index]
            + variant.alternate_bases
            + reference_dna[variant_index + len(variant.reference_bases) :]
        )

        return GenomicWindow(
            chromosome=variant.chromosome,
            start=start,
            end=end,
            variant_index=variant_index,
            reference_dna=reference_dna,
            alternate_dna=alternate_dna,
        )

    def analyse_gene_isoforms(
        self,
        gene_symbol: str,
        variant: VariantInput,
        *,
        vep_result: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """
        Retrieve every translated Ensembl transcript and reconstruct ALT proteins
        only when the variant directly maps to the annotated CDS.
        """
        gene = self.lookup_gene(gene_symbol, expand=True)
        gene_id = strip_version(gene.get("id"))
        canonical_transcript = strip_version(gene.get("canonical_transcript"))
        if vep_result is None:
            vep_result = self.annotate_variant(variant)

        consequences: dict[str, dict[str, Any]] = {}
        for item in vep_result.get("transcript_consequences", []):
            if strip_version(item.get("gene_id")) != gene_id:
                continue
            transcript_id = strip_version(item.get("transcript_id"))
            if transcript_id:
                consequences[transcript_id] = item

        rows: list[dict[str, Any]] = []
        translated_transcripts = []
        for transcript in gene.get("Transcript", []):
            translation = transcript.get("Translation")
            if not translation:
                continue
            transcript_id = strip_version(transcript.get("id"))
            protein_id = strip_version(translation.get("id"))
            if transcript_id and protein_id:
                translated_transcripts.append((transcript, transcript_id, protein_id))

        if not translated_transcripts:
            raise ValueError(f"No translated Ensembl transcripts found for {gene_symbol}.")

        for transcript, transcript_id, protein_id in translated_transcripts:
            consequence = consequences.get(transcript_id)
            strand_number = int(transcript.get("strand", gene.get("strand", 1)))
            reference_protein = self.get_sequence(protein_id, "protein")

            row: dict[str, Any] = {
                "gene_name": str(gene_symbol).upper(),
                "gene_id": gene_id,
                "transcript_id": transcript_id,
                "transcript_version": transcript.get("version"),
                "protein_id": protein_id,
                "protein_version": translation_version(translation),
                "transcript_biotype": transcript.get("biotype"),
                "gene_strand": "+" if int(gene.get("strand", 1)) == 1 else "-",
                "is_ensembl_canonical": transcript_id == canonical_transcript,
                "mane_select": False,
                "mane_plus_clinical": False,
                "appris": None,
                "ccds": None,
                "consequence_terms": "",
                "direct_cds_overlap": False,
                "cds_position": None,
                "protein_position": None,
                "codon_change": None,
                "amino_acid_change": None,
                "hgvsc": None,
                "hgvsp": None,
                "reference_protein": reference_protein,
                "alternate_protein": None,
                "protein_changed": None,
                "alternate_status": "not_resolved_no_direct_cds_change",
                "mapping_error": None,
            }

            if consequence is None:
                row["alternate_status"] = "no_vep_consequence_for_transcript"
                rows.append(row)
                continue

            terms = consequence.get("consequence_terms", []) or []
            row.update(
                {
                    "mane_select": bool(consequence.get("mane_select")),
                    "mane_plus_clinical": bool(consequence.get("mane_plus_clinical")),
                    "appris": consequence.get("appris"),
                    "ccds": consequence.get("ccds"),
                    "consequence_terms": ",".join(map(str, terms)),
                    "codon_change": consequence.get("codons"),
                    "amino_acid_change": consequence.get("amino_acids"),
                    "hgvsc": consequence.get("hgvsc"),
                    "hgvsp": consequence.get("hgvsp"),
                    "protein_position": consequence.get("protein_start"),
                }
            )

            cds_start = consequence.get("cds_start")
            cds_end = consequence.get("cds_end")
            if cds_start is None:
                rows.append(row)
                continue

            row["direct_cds_overlap"] = True
            try:
                reference_cds = self.get_sequence(transcript_id, "cds")
                transcript_ref = variant.reference_bases
                transcript_alt = variant.alternate_bases
                if strand_number == -1:
                    transcript_ref = reverse_complement(transcript_ref)
                    transcript_alt = reverse_complement(transcript_alt)

                cds_start_i = int(cds_start)
                cds_end_i = int(cds_end if cds_end is not None else cds_start)
                cds_index = cds_start_i - 1
                observed = reference_cds[cds_index : cds_index + len(transcript_ref)]
                if observed != transcript_ref:
                    raise ValueError(
                        f"CDS reference mismatch: expected {transcript_ref}, observed {observed}"
                    )

                alternate_cds = (
                    reference_cds[:cds_index]
                    + transcript_alt
                    + reference_cds[cds_index + len(transcript_ref) :]
                )
                alternate_protein = translate_cds(alternate_cds)
                protein_changed = alternate_protein != reference_protein

                row.update(
                    {
                        "cds_position": (
                            str(cds_start_i)
                            if cds_start_i == cds_end_i
                            else f"{cds_start_i}-{cds_end_i}"
                        ),
                        "alternate_protein": alternate_protein,
                        "protein_changed": protein_changed,
                        "alternate_status": (
                            "resolved_changed"
                            if protein_changed
                            else "resolved_same_as_reference"
                        ),
                    }
                )
                if "stop_lost" in terms:
                    row["alternate_status"] = (
                        "partial_stop_lost_extension_not_reconstructed"
                    )
            except Exception as exc:  # Preserve partial results per transcript.
                row["mapping_error"] = str(exc)
                row["alternate_status"] = "mapping_failed"
                row["alternate_protein"] = None
                row["protein_changed"] = None

            rows.append(row)

        summary_df = pd.DataFrame(rows)
        summary_df = _assign_reference_isoforms(summary_df)
        summary_df = _assign_alternate_isoforms(summary_df)
        summary_df["reference_aa_length"] = summary_df["reference_protein"].str.len()
        summary_df["alternate_aa_length"] = summary_df["alternate_protein"].map(
            lambda x: len(x) if isinstance(x, str) else None
        )

        summary_df = summary_df.sort_values(
            by=["is_ensembl_canonical", "mane_select", "reference_isoform", "transcript_id"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

        metadata = {
            "gene_name": str(gene_symbol).upper(),
            "gene_id": gene_id,
            "chromosome": str(gene.get("seq_region_name")),
            "start": int(gene.get("start")),
            "end": int(gene.get("end")),
            "strand": "+" if int(gene.get("strand", 1)) == 1 else "-",
            "most_severe_consequence": vep_result.get("most_severe_consequence"),
            "translated_transcripts": len(summary_df),
            "unique_reference_isoforms": summary_df["reference_isoform"].nunique(),
            "variant_resolved_transcripts": int(summary_df["alternate_protein"].notna().sum()),
        }
        return metadata, summary_df


def translation_version(translation: dict[str, Any]) -> Any:
    return translation.get("version")


def _sequence_hash(sequence: str | None) -> str | None:
    if not isinstance(sequence, str):
        return None
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _assign_reference_isoforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hashes = df["reference_protein"].map(_sequence_hash)
    canonical_hashes = set(
        hashes[df["is_ensembl_canonical"].fillna(False)].dropna().tolist()
    )
    mane_hashes = set(hashes[df["mane_select"].fillna(False)].dropna().tolist())

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, value in hashes.items():
        if value is not None:
            groups[value].append(idx)

    ordered = sorted(
        groups,
        key=lambda h: (
            h not in canonical_hashes,
            h not in mane_hashes,
            -len(str(df.loc[groups[h][0], "reference_protein"])),
            str(df.loc[groups[h][0], "transcript_id"]),
        ),
    )
    mapping = {sequence_hash: f"ISOFORM_{i}" for i, sequence_hash in enumerate(ordered, 1)}
    df["reference_isoform"] = hashes.map(mapping)
    return df


def _assign_alternate_isoforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hashes = df["alternate_protein"].map(_sequence_hash)
    unique_hashes = [h for h in dict.fromkeys(hashes.dropna().tolist())]
    unique_hashes.sort(
        key=lambda h: (
            -len(str(df.loc[hashes[hashes == h].index[0], "alternate_protein"])),
            str(df.loc[hashes[hashes == h].index[0], "transcript_id"]),
        )
    )
    mapping = {sequence_hash: f"ALT_ISOFORM_{i}" for i, sequence_hash in enumerate(unique_hashes, 1)}
    df["alternate_isoform"] = hashes.map(mapping)
    return df
