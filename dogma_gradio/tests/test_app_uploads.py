from pathlib import Path

from app import _variant_count, review_uploads


def test_variant_count_handles_vcf_metadata(tmp_path: Path):
    vcf = tmp_path / "variants.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tG\t.\tPASS\t.\n"
        "1\t20\t.\tC\tT\t.\tPASS\t.\n"
    )

    assert _variant_count(vcf) == 2


def test_review_uploads_requires_both_labelled_cohorts(tmp_path: Path):
    pathogenic = tmp_path / "pathogenic.csv"
    pathogenic.write_text("chrom,pos,ref,alt\n1,10,A,G\n")

    result = review_uploads([str(pathogenic)], None)

    assert "Almost there" in result
    assert "Add at least one benign file" in result


def test_review_uploads_summarises_ready_files(tmp_path: Path):
    pathogenic = tmp_path / "pathogenic.txt"
    benign = tmp_path / "benign.csv"
    pathogenic.write_text("variant\nrs1\nrs2\n")
    benign.write_text("variant\nrs3\n")

    result = review_uploads([str(pathogenic)], [str(benign)])

    assert "Files are ready" in result
    assert "2 variant rows" in result
    assert "1 variant row" in result
