"""Exact review-collection indexing for response-window evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER
from reader.domains.review import (
    ReviewCollectionIndex,
    ReviewEntity,
    ReviewEntityKind,
    ReviewExperiment,
    ReviewOccurrence,
)

_ROW_ID_COLUMNS = {"experiment_id", "design_id", "reduction_id", "is_reference"}
_SHARED_REDUCTION_COLUMNS = (
    "reference_design_id",
    "reduction_method",
    "response_basis",
    "reduction_role",
    "replicate_stat",
    "bootstrap_samples",
    "confidence_level",
    "window_start_event_h",
    "window_end_event_h",
)


def response_window_review_collection(
    designs: pd.DataFrame,
    *,
    experiment_ids: Sequence[str],
    experiment_titles: Mapping[str, str],
    review_collection_id: str,
    review_collection_label: str,
) -> ReviewCollectionIndex:
    """Build navigation from exact primary, non-reference Reader design rows."""

    required = _ROW_ID_COLUMNS | {"reduction_role"}
    _require_columns(designs, required, context="response-window review collection")
    declared_ids = tuple(_required_text(value, field="experiment_id") for value in experiment_ids)
    if not declared_ids:
        raise ValueError("response-window review collection requires declared experiments.")
    if len(set(declared_ids)) != len(declared_ids):
        raise ValueError("response-window review collection contains duplicate experiment IDs.")
    title_ids = set(experiment_titles)
    if title_ids != set(declared_ids):
        raise ValueError(
            "response-window experiment titles must match the declared review collection exactly: "
            f"missing={sorted(set(declared_ids) - title_ids)}, extra={sorted(title_ids - set(declared_ids))}."
        )
    observed_ids = set(designs["experiment_id"].astype(str))
    if observed_ids != set(declared_ids):
        raise ValueError(
            "response-window rows must match the declared review collection exactly: "
            f"missing={sorted(set(declared_ids) - observed_ids)}, extra={sorted(observed_ids - set(declared_ids))}."
        )
    if not is_bool_dtype(designs["is_reference"]):
        raise ValueError("response-window is_reference must be a boolean column.")

    primary = designs.loc[designs["reduction_role"].astype(str).eq("primary")].copy()
    if primary.empty:
        raise ValueError("response-window review collection contains no primary reduction rows.")
    duplicate = primary.duplicated(["experiment_id", "design_id"], keep=False)
    if duplicate.any():
        keys = primary.loc[duplicate, ["experiment_id", "design_id"]].astype(str).drop_duplicates()
        raise ValueError(
            "response-window primary review requires one row per experiment and Reader design: "
            f"{keys.to_dict('records')}."
        )
    entity_rows = primary.loc[~primary["is_reference"]].copy()
    if entity_rows.empty:
        raise ValueError("response-window review collection contains no non-reference Reader designs.")

    entity_ids = sorted(entity_rows["design_id"].astype(str).unique())
    experiments = tuple(
        ReviewExperiment(
            experiment_id=experiment_id,
            display_title=_required_text(experiment_titles[experiment_id], field="experiment title"),
        )
        for experiment_id in declared_ids
    )
    entities = tuple(ReviewEntity(entity_id=entity_id, display_label=entity_id) for entity_id in entity_ids)
    occurrence_rows = entity_rows.loc[:, ["experiment_id", "design_id"]].sort_values(
        ["experiment_id", "design_id"], kind="mergesort"
    )
    occurrences = tuple(
        ReviewOccurrence(entity_id=str(row.design_id), experiment_id=str(row.experiment_id))
        for row in occurrence_rows.itertuples(index=False)
    )
    return ReviewCollectionIndex(
        review_collection_id=review_collection_id,
        review_collection_label=review_collection_label,
        entity_kind=ReviewEntityKind(kind_id="reader.design_id", selector_label="Reader design"),
        experiments=experiments,
        entities=entities,
        occurrences=occurrences,
    )


def cross_experiment_design_rows(
    designs: pd.DataFrame,
    *,
    design_id: str,
    reduction_id: str,
) -> pd.DataFrame:
    """Select and validate one exact design and reduction across experiments."""

    exact_design_id = _required_text(design_id, field="Reader design")
    exact_reduction_id = _required_text(reduction_id, field="response summary")
    required = _ROW_ID_COLUMNS | set(_SHARED_REDUCTION_COLUMNS) | _component_columns()
    _require_columns(designs, required, context="cross-experiment response-window review")
    selected = designs.loc[
        designs["design_id"].astype(str).eq(exact_design_id)
        & designs["reduction_id"].astype(str).eq(exact_reduction_id)
    ].copy()
    if selected["experiment_id"].astype(str).nunique() < 2:
        raise ValueError(
            f"Reader design {exact_design_id!r} requires at least two experiments for cross-experiment review."
        )
    if selected["experiment_id"].astype(str).duplicated().any():
        raise ValueError(
            "cross-experiment response-window review requires one row per experiment for "
            f"Reader design {exact_design_id!r} and response summary {exact_reduction_id!r}."
        )
    if not is_bool_dtype(selected["is_reference"]):
        raise ValueError("response-window is_reference must be a boolean column.")
    if selected["is_reference"].any():
        raise ValueError("cross-experiment review requires a non-reference Reader design.")

    drift = [column for column in _SHARED_REDUCTION_COLUMNS if selected[column].nunique(dropna=False) != 1]
    if drift:
        raise ValueError(f"cross-experiment rows must use shared reduction semantics; drifted columns: {drift}.")
    _validate_numeric_evidence(selected)
    return selected.sort_values("experiment_id", kind="mergesort").reset_index(drop=True)


def _validate_numeric_evidence(selected: pd.DataFrame) -> None:
    numeric_columns = sorted(_component_columns() - {f"n{state}" for state in STATE_ORDER})
    numeric = selected.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("cross-experiment response-window evidence must be finite.")
    confidence = float(selected["confidence_level"].iloc[0])
    if not 0.0 < confidence < 1.0:
        raise ValueError("cross-experiment confidence level must lie strictly between zero and one.")
    for state in STATE_ORDER:
        count = pd.to_numeric(selected[f"n{state}"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(count).all() or np.any(count < 1) or np.any(count != np.floor(count)):
            raise ValueError(f"cross-experiment replicate counts must be positive integers for state {state}.")
        for prefix in ("r", "b"):
            value = selected[f"{prefix}{state}"].to_numpy(dtype=float)
            low = selected[f"{prefix}{state}_ci_low"].to_numpy(dtype=float)
            high = selected[f"{prefix}{state}_ci_high"].to_numpy(dtype=float)
            if np.any(low > value) or np.any(value > high):
                raise ValueError(f"{prefix}{state} interval does not contain its published summary.")
            event = selected[f"{prefix}{state}_event_half_range"].to_numpy(dtype=float)
            if np.any(event < 0.0):
                raise ValueError(f"{prefix}{state} event-time sensitivity must be nonnegative.")


def _component_columns() -> set[str]:
    columns: set[str] = set()
    for state in STATE_ORDER:
        columns.add(f"n{state}")
        for prefix in ("r", "b"):
            columns.update(
                {
                    f"{prefix}{state}",
                    f"{prefix}{state}_ci_low",
                    f"{prefix}{state}_ci_high",
                    f"{prefix}{state}_event_half_range",
                }
            )
    return columns


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}.")


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace.")
    return value


__all__ = ["cross_experiment_design_rows", "response_window_review_collection"]
