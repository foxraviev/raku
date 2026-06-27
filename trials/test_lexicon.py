from __future__ import annotations

from raku.slip import lexicon


def test_vocabulary_cardinalities() -> None:
    assert len(lexicon.ANATOMICAL) == 12
    assert len(lexicon.PATHOLOGICAL) == 45
    assert len(lexicon.SEVERITY) == 9
    assert lexicon.total_concepts() == 66


def test_grounding_edges_are_in_range() -> None:
    edges = lexicon.grounding_edges()
    assert edges
    for k, j in edges:
        assert 0 <= k < len(lexicon.PATHOLOGICAL)
        assert 0 <= j < len(lexicon.ANATOMICAL)


def test_every_pathology_has_an_anchor() -> None:
    grounded = {k for k, _ in lexicon.grounding_edges()}
    assert grounded == set(range(len(lexicon.PATHOLOGICAL)))


def test_indicator_matches_class_count() -> None:
    for name in ("odir5k", "rfmid", "jsiec"):
        rows = lexicon.indicator_matrix(name)
        assert len(rows) == len(lexicon.class_names(name))
        assert all(len(r) == lexicon.total_concepts() for r in rows)
        assert len(lexicon.class_weights(name)) == len(rows)


def test_normal_class_has_no_pathology() -> None:
    rows = lexicon.indicator_matrix("odir5k")
    normal = rows[0]
    assert sum(normal[12:57]) == 0.0
    assert sum(normal[:12]) == 12.0
