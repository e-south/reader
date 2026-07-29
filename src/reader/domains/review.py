"""Assay-neutral identity and navigation contracts for review collections."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewEntityKind:
    """Stable identity and readable selector label for a review entity kind."""

    kind_id: str
    selector_label: str


@dataclass(frozen=True)
class ReviewExperiment:
    """Stable Reader experiment identity plus presentation metadata."""

    experiment_id: str
    display_title: str


@dataclass(frozen=True)
class ReviewEntity:
    """One exact entity available in a review collection."""

    entity_id: str
    display_label: str


@dataclass(frozen=True)
class ReviewOccurrence:
    """Exact membership of an entity in one Reader experiment."""

    entity_id: str
    experiment_id: str


@dataclass(frozen=True)
class ReviewCollectionIndex:
    """Validated navigation index over one explicitly declared review collection."""

    review_collection_id: str
    review_collection_label: str
    entity_kind: ReviewEntityKind
    experiments: tuple[ReviewExperiment, ...]
    entities: tuple[ReviewEntity, ...]
    occurrences: tuple[ReviewOccurrence, ...]

    def __post_init__(self) -> None:
        _required_text(self.review_collection_id, field="review_collection_id")
        _required_text(self.review_collection_label, field="review_collection_label")
        _required_text(self.entity_kind.kind_id, field="kind_id")
        _required_text(self.entity_kind.selector_label, field="entity selector label")
        if not self.experiments:
            raise ValueError("review collection requires at least one experiment.")
        if not self.entities:
            raise ValueError("review collection requires at least one entity.")

        experiment_ids = []
        for experiment in self.experiments:
            experiment_ids.append(_required_text(experiment.experiment_id, field="experiment_id"))
            _required_text(experiment.display_title, field="experiment display title")
        _reject_duplicates(experiment_ids, field="experiment IDs")

        entity_ids = []
        for entity in self.entities:
            entity_ids.append(_required_text(entity.entity_id, field="entity_id"))
            _required_text(entity.display_label, field="entity display label")
        _reject_duplicates(entity_ids, field="entity IDs")

        experiment_id_set = set(experiment_ids)
        entity_id_set = set(entity_ids)
        occurrence_keys: list[tuple[str, str]] = []
        for occurrence in self.occurrences:
            entity_id = _required_text(occurrence.entity_id, field="occurrence entity_id")
            experiment_id = _required_text(occurrence.experiment_id, field="occurrence experiment_id")
            if entity_id not in entity_id_set or experiment_id not in experiment_id_set:
                raise ValueError(
                    "review occurrences must reference declared entities and experiments: "
                    f"{(entity_id, experiment_id)!r} is undeclared."
                )
            occurrence_keys.append((entity_id, experiment_id))
        _reject_duplicates(occurrence_keys, field="entity occurrences")

    def multi_experiment_entity_ids(self, *, minimum: int = 2) -> tuple[str, ...]:
        """Return exact entities observed in at least ``minimum`` experiments."""

        if minimum < 2:
            raise ValueError("multi-experiment review requires at least two experiments.")
        counts = Counter(occurrence.entity_id for occurrence in self.occurrences)
        return tuple(entity.entity_id for entity in self._ordered_entities() if counts[entity.entity_id] >= minimum)

    def multi_experiment_entity_options(self) -> dict[str, str]:
        """Map disambiguated display labels to stable multi-experiment entity IDs."""

        allowed_ids = set(self.multi_experiment_entity_ids())
        entities = [entity for entity in self._ordered_entities() if entity.entity_id in allowed_ids]
        label_counts = Counter(entity.display_label for entity in entities)
        pairs = [
            (
                (
                    entity.display_label
                    if label_counts[entity.display_label] == 1
                    else f"{entity.display_label} · {entity.entity_id}"
                ),
                entity.entity_id,
            )
            for entity in entities
        ]
        options = dict(pairs)
        if len(options) != len(pairs):
            raise ValueError(
                f"{self.entity_kind.selector_label} selector labels remain ambiguous after stable-ID disambiguation."
            )
        return options

    def experiments_for_entity(self, entity_id: str) -> tuple[ReviewExperiment, ...]:
        """Return collection-ordered experiments for one exact multi-experiment entity."""

        exact_id = _required_text(entity_id, field=self.entity_kind.selector_label)
        entity_ids = {entity.entity_id for entity in self.entities}
        if exact_id not in entity_ids:
            raise ValueError(f"unknown {self.entity_kind.selector_label}: {exact_id!r}.")
        observed_ids = {occurrence.experiment_id for occurrence in self.occurrences if occurrence.entity_id == exact_id}
        experiments = tuple(experiment for experiment in self.experiments if experiment.experiment_id in observed_ids)
        if len(experiments) < 2:
            raise ValueError(
                f"{self.entity_kind.selector_label} {exact_id!r} requires at least two experiments for review; "
                f"found {len(experiments)}."
            )
        return experiments

    def entities_for_experiment(self, experiment_id: str) -> tuple[ReviewEntity, ...]:
        """Return exact entities observed in one declared experiment."""

        exact_id = _required_text(experiment_id, field="experiment_id")
        if exact_id not in {experiment.experiment_id for experiment in self.experiments}:
            raise ValueError(f"unknown experiment_id: {exact_id!r}.")
        observed_ids = {occurrence.entity_id for occurrence in self.occurrences if occurrence.experiment_id == exact_id}
        return tuple(entity for entity in self._ordered_entities() if entity.entity_id in observed_ids)

    def _ordered_entities(self) -> tuple[ReviewEntity, ...]:
        return tuple(sorted(self.entities, key=lambda item: (item.display_label.casefold(), item.entity_id)))


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    text = value.strip()
    if text != value:
        raise ValueError(f"{field} must not contain leading or trailing whitespace.")
    return text


def _reject_duplicates(values: list[object], *, field: str) -> None:
    duplicates = sorted({value for value, count in Counter(values).items() if count > 1}, key=str)
    if duplicates:
        raise ValueError(f"review collection contains duplicate {field}: {duplicates}.")


def retained_review_selection(available_ids: Iterable[str], *, preferred_id: str | None) -> str:
    """Keep an exact review entity selected when it is available in a new scope."""

    exact_ids = tuple(_required_text(value, field="entity_id") for value in available_ids)
    if not exact_ids:
        raise ValueError("review selection requires at least one exact entity ID.")
    _reject_duplicates(list(exact_ids), field="entity IDs")
    if preferred_id is None:
        return exact_ids[0]
    preferred = _required_text(preferred_id, field="preferred entity_id")
    return preferred if preferred in exact_ids else exact_ids[0]


def retained_review_option_key(options: Mapping[str, str], *, preferred_id: str | None) -> str:
    """Return the display key for a retained stable ID in a mapped selector."""

    selected_id = retained_review_selection(options.values(), preferred_id=preferred_id)
    matching_keys = [key for key, value in options.items() if value == selected_id]
    if len(matching_keys) != 1:
        raise ValueError("review selector values must map one display key to each exact entity ID.")
    return _required_text(matching_keys[0], field="entity selector key")


__all__ = [
    "ReviewCollectionIndex",
    "ReviewEntity",
    "ReviewEntityKind",
    "ReviewExperiment",
    "ReviewOccurrence",
    "retained_review_option_key",
    "retained_review_selection",
]
