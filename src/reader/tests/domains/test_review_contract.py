from __future__ import annotations

from pathlib import Path

import pytest

from reader.domains.review import (
    ReviewCollectionIndex,
    ReviewEntity,
    ReviewEntityKind,
    ReviewExperiment,
    ReviewOccurrence,
    retained_review_option_key,
    retained_review_selection,
)


def test_review_collection_exposes_multi_experiment_entities_and_exact_experiments() -> None:
    index = _index()

    assert index.multi_experiment_entity_ids() == ("design_alpha",)
    assert index.multi_experiment_entity_options() == {"Shared design": "design_alpha"}
    assert [item.experiment_id for item in index.experiments_for_entity("design_alpha")] == [
        "experiment_2",
        "experiment_1",
    ]
    assert [item.entity_id for item in index.entities_for_experiment("experiment_1")] == [
        "design_beta",
        "design_alpha",
    ]


def test_review_collection_disambiguates_duplicate_display_labels() -> None:
    index = ReviewCollectionIndex(
        review_collection_id="review_1",
        review_collection_label="Response review",
        entity_kind=ReviewEntityKind(kind_id="reader.design_id", selector_label="Reader design"),
        experiments=(
            ReviewExperiment("experiment_1", "Run one"),
            ReviewExperiment("experiment_2", "Run two"),
        ),
        entities=(
            ReviewEntity("design_alpha", "Shared label"),
            ReviewEntity("design_beta", "Shared label"),
        ),
        occurrences=(
            ReviewOccurrence("design_alpha", "experiment_1"),
            ReviewOccurrence("design_alpha", "experiment_2"),
            ReviewOccurrence("design_beta", "experiment_1"),
            ReviewOccurrence("design_beta", "experiment_2"),
        ),
    )

    assert index.multi_experiment_entity_options() == {
        "Shared label · design_alpha": "design_alpha",
        "Shared label · design_beta": "design_beta",
    }


def test_review_collection_rejects_ambiguous_generated_selector_labels() -> None:
    index = ReviewCollectionIndex(
        review_collection_id="review_1",
        review_collection_label="Response review",
        entity_kind=ReviewEntityKind(kind_id="reader.design_id", selector_label="Reader design"),
        experiments=(
            ReviewExperiment("experiment_1", "Run one"),
            ReviewExperiment("experiment_2", "Run two"),
        ),
        entities=(
            ReviewEntity("a", "Shared"),
            ReviewEntity("b", "Shared"),
            ReviewEntity("c", "Shared · a"),
        ),
        occurrences=tuple(
            ReviewOccurrence(entity_id, experiment_id)
            for entity_id in ("a", "b", "c")
            for experiment_id in ("experiment_1", "experiment_2")
        ),
    )

    with pytest.raises(ValueError, match="selector labels remain ambiguous"):
        index.multi_experiment_entity_options()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("review_collection_id", " ", "review_collection_id"),
        ("entity_kind", ReviewEntityKind(kind_id=" ", selector_label="Reader design"), "kind_id"),
        (
            "experiments",
            (ReviewExperiment("experiment_1", "Run one"), ReviewExperiment("experiment_1", "Run again")),
            "duplicate experiment IDs",
        ),
        (
            "entities",
            (ReviewEntity("design_alpha", "A"), ReviewEntity("design_alpha", "B")),
            "duplicate entity IDs",
        ),
        (
            "occurrences",
            (
                ReviewOccurrence("design_alpha", "experiment_1"),
                ReviewOccurrence("design_alpha", "experiment_1"),
            ),
            "duplicate entity occurrences",
        ),
    ],
)
def test_review_collection_rejects_invalid_identity_contracts(field: str, replacement: object, message: str) -> None:
    values = _index_values()
    values[field] = replacement

    with pytest.raises(ValueError, match=message):
        ReviewCollectionIndex(**values)


@pytest.mark.parametrize(
    "occurrence",
    [
        ReviewOccurrence("unknown_design", "experiment_1"),
        ReviewOccurrence("design_alpha", "unknown_experiment"),
    ],
)
def test_review_collection_rejects_unknown_occurrence_references(occurrence: ReviewOccurrence) -> None:
    values = _index_values()
    values["occurrences"] = (*values["occurrences"], occurrence)

    with pytest.raises(ValueError, match="undeclared"):
        ReviewCollectionIndex(**values)


def test_review_collection_rejects_single_experiment_comparison() -> None:
    index = _index()

    with pytest.raises(ValueError, match="at least two experiments"):
        index.experiments_for_entity("design_beta")


def test_retained_review_selection_preserves_focus_only_when_available() -> None:
    assert retained_review_selection(("pDual-10", "ES28"), preferred_id="ES28") == "ES28"
    assert retained_review_selection(("pDual-10", "ES30"), preferred_id="ES28") == "pDual-10"
    assert retained_review_selection(("pDual-10", "ES28"), preferred_id=None) == "pDual-10"


def test_retained_review_selection_rejects_empty_or_ambiguous_options() -> None:
    with pytest.raises(ValueError, match="at least one exact entity"):
        retained_review_selection((), preferred_id="ES28")
    with pytest.raises(ValueError, match="duplicate entity IDs"):
        retained_review_selection(("ES28", "ES28"), preferred_id="ES28")


def test_retained_review_option_key_preserves_disambiguated_labels() -> None:
    options = {
        "Shared label · design_a": "design_a",
        "Shared label · design_b": "design_b",
    }

    assert retained_review_option_key(options, preferred_id="design_b") == "Shared label · design_b"
    assert retained_review_option_key(options, preferred_id="missing") == "Shared label · design_a"


def test_review_collection_uses_exact_entity_ids() -> None:
    index = _index()

    with pytest.raises(ValueError, match="unknown Reader design"):
        index.experiments_for_entity("design")


def test_review_contract_stays_small_and_assay_neutral() -> None:
    path = Path(__file__).resolve().parents[2] / "domains" / "review.py"
    source = path.read_text(encoding="utf-8")

    assert len(source.splitlines()) <= 200
    for forbidden in ("pandas", "marimo", "response_window", "sfxi", "rmf", "campaign"):
        assert forbidden not in source.lower()


def _index() -> ReviewCollectionIndex:
    return ReviewCollectionIndex(**_index_values())


def _index_values() -> dict[str, object]:
    return {
        "review_collection_id": "review_1",
        "review_collection_label": "Response review",
        "entity_kind": ReviewEntityKind(kind_id="reader.design_id", selector_label="Reader design"),
        "experiments": (
            ReviewExperiment("experiment_2", "Run two"),
            ReviewExperiment("experiment_1", "Run one"),
        ),
        "entities": (
            ReviewEntity("design_beta", "Other design"),
            ReviewEntity("design_alpha", "Shared design"),
        ),
        "occurrences": (
            ReviewOccurrence("design_alpha", "experiment_1"),
            ReviewOccurrence("design_alpha", "experiment_2"),
            ReviewOccurrence("design_beta", "experiment_1"),
        ),
    }
