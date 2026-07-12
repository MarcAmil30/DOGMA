from types import MappingProxyType, SimpleNamespace

from dogma.alphagenome_service import _recommended_scorer_map


def test_accepts_immutable_recommended_scorer_mapping():
    scorer = object()
    module = SimpleNamespace(
        RECOMMENDED_VARIANT_SCORERS=MappingProxyType({"RNA_SEQ": scorer})
    )

    assert _recommended_scorer_map(module, object()) == {"RNA_SEQ": scorer}


def test_recommended_helper_receives_numeric_organism_value():
    scorer = SimpleNamespace(requested_output=SimpleNamespace(name="RNA_SEQ"))
    received = []

    def get_recommended_scorers(organism):
        received.append(organism)
        return [scorer]

    module = SimpleNamespace(get_recommended_scorers=get_recommended_scorers)
    organism = SimpleNamespace(value=9606)

    assert _recommended_scorer_map(module, organism) == {"RNA_SEQ": scorer}
    assert received == [9606]
