"""Validated context for one cross-experiment response-window figure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER

from .review_collection import cross_experiment_design_rows
from .review_experiment_labels import compact_experiment_plot_labels
from .visual_labels import channels, state_labels


@dataclass(frozen=True)
class CrossExperimentContext:
    design_id: str
    reduction_id: str
    reference_id: str
    state: str
    state_label: str
    experiment_order: tuple[str, ...]
    experiment_labels: Mapping[str, str]
    plot_experiment_labels: Mapping[str, str]
    event_uncertainty: Mapping[str, float]
    confidence_level: float


def prepare_cross_experiment_context(
    *,
    selected: pd.DataFrame,
    state: str,
    experiment_labels: Mapping[str, str],
    events: pd.DataFrame,
    display: dict[str, object],
) -> tuple[pd.DataFrame, CrossExperimentContext]:
    if state not in STATE_ORDER:
        raise ValueError(f"unknown response-window state: {state!r}; expected one of {list(STATE_ORDER)}.")
    if selected.empty or "design_id" not in selected or selected["design_id"].astype(str).nunique() != 1:
        raise ValueError("cross-experiment figure requires exactly one Reader design.")
    if "reduction_id" not in selected or selected["reduction_id"].astype(str).nunique() != 1:
        raise ValueError("cross-experiment figure requires exactly one response summary.")
    design_id = str(selected["design_id"].iloc[0])
    reduction_id = str(selected["reduction_id"].iloc[0])
    selected = cross_experiment_design_rows(selected, design_id=design_id, reduction_id=reduction_id)
    experiment_order, labels = _experiment_order(selected, experiment_labels)
    selected = selected.set_index("experiment_id").loc[list(experiment_order)].reset_index()

    display_channels = channels(display)
    reference_id = str(selected["reference_design_id"].iloc[0])
    if reference_id != display_channels["reference_design_id"]:
        raise ValueError("cross-experiment reference identity disagrees with the display contract.")
    context = CrossExperimentContext(
        design_id=design_id,
        reduction_id=reduction_id,
        reference_id=reference_id,
        state=state,
        state_label=state_labels(display)[state],
        experiment_order=experiment_order,
        experiment_labels=labels,
        plot_experiment_labels=compact_experiment_plot_labels(experiment_order, labels),
        event_uncertainty=_event_uncertainty(events, experiment_order),
        confidence_level=float(selected["confidence_level"].iloc[0]),
    )
    return selected, context


def _experiment_order(
    selected: pd.DataFrame,
    experiment_labels: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[str, str]]:
    selected_ids = set(selected["experiment_id"].astype(str))
    if set(experiment_labels) != selected_ids:
        raise ValueError("cross-experiment labels must match selected experiments exactly.")
    order = tuple(str(experiment_id) for experiment_id in experiment_labels)
    labels = {experiment_id: str(experiment_labels[experiment_id]).strip() for experiment_id in order}
    if any(not label for label in labels.values()):
        raise ValueError("cross-experiment labels must be non-empty.")
    if len(set(labels.values())) != len(labels):
        raise ValueError("cross-experiment labels must be unique.")
    return order, labels


def _event_uncertainty(events: pd.DataFrame, experiment_order: tuple[str, ...]) -> dict[str, float]:
    required = {"experiment_id", "event_time_uncertainty_h"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"cross-experiment events are missing columns: {missing}.")
    result: dict[str, float] = {}
    for experiment_id in experiment_order:
        selected = events.loc[events["experiment_id"].astype(str).eq(experiment_id)]
        if len(selected) != 1:
            raise ValueError(f"experiment {experiment_id!r} must have exactly one event record.")
        uncertainty = float(selected["event_time_uncertainty_h"].iloc[0])
        if not np.isfinite(uncertainty) or uncertainty < 0.0:
            raise ValueError(f"experiment {experiment_id!r} has invalid event-time uncertainty.")
        result[experiment_id] = uncertainty
    return result


__all__ = ["CrossExperimentContext", "prepare_cross_experiment_context"]
