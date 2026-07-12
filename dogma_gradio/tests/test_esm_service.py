import sys
from types import SimpleNamespace

import pandas as pd

from dogma.esm_service import AA_ORDER, run_esm_for_isoforms


class _Input:
    def __init__(self, *, sequences):
        self.sequences = sequences


class _Config:
    def __init__(self, **kwargs):
        self.options = kwargs


def _fake_proto_tools(run_function):
    return SimpleNamespace(
        ESM2SampleConfig=_Config,
        ESM2SampleInput=_Input,
        run_esm2_sample=run_function,
    )


def _isoform_row(reference: str, alternate: str | None) -> dict[str, object]:
    return {
        "gene_name": "TEST",
        "reference_isoform": "ISOFORM_1",
        "alternate_isoform": "ALT_ISOFORM_1",
        "transcript_id": "ENST_TEST",
        "protein_id": "ENSP_TEST",
        "is_ensembl_canonical": True,
        "consequence_terms": "missense_variant",
        "protein_changed": reference != alternate,
        "alternate_status": "resolved_changed",
        "reference_aa_length": len(reference),
        "alternate_aa_length": len(alternate) if alternate is not None else None,
        "reference_protein": reference,
        "alternate_protein": alternate,
    }


def test_scores_only_masked_changed_position(monkeypatch):
    calls = []

    def fake_run(inputs, config):
        calls.append(inputs.sequences)
        assert inputs.sequences == ["M_C"]
        logits = [[[0.0] * len(AA_ORDER) for _ in sequence] for sequence in inputs.sequences]
        # At position 2, REF=A has logit 2 and ALT=C has logit 1. The shared
        # log-softmax normaliser cancels, so ALT minus REF is exactly -1.
        logits[0][1][AA_ORDER.index("A")] = 2.0
        logits[0][1][AA_ORDER.index("C")] = 1.0
        return SimpleNamespace(logits=logits)

    monkeypatch.setitem(sys.modules, "proto_tools", _fake_proto_tools(fake_run))
    result = run_esm_for_isoforms(pd.DataFrame([_isoform_row("MAC", "MCC")]))

    assert calls == [["M_C"]]
    assert len(result) == 1
    assert result.loc[0, "mutation"] == "A2C"
    assert result.loc[0, "esm_status"] == "masked_position_scored"
    assert result.loc[0, "delta_position_log_probability_alt_minus_ref"] == -1.0
    assert "reference_avg_log_likelihood" not in result.columns


def test_skips_length_changing_protein_without_model_call(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("ESM2 should not run for a protein length change")

    monkeypatch.setitem(sys.modules, "proto_tools", _fake_proto_tools(unexpected_call))
    result = run_esm_for_isoforms(pd.DataFrame([_isoform_row("MAC", "MA")]))

    assert len(result) == 1
    assert result.loc[0, "esm_status"] == "not_scored_protein_length_change"
