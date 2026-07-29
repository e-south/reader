"""Manifest-backed source loading for response-window analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import EventSpec, ResponseWindowSourceSpec

STATE_ORDER = ("00", "10", "01", "11")
ANNOTATED_CONTRACT = "plate_reader.annotated.v1"


@dataclass(frozen=True)
class EventInterval:
    experiment_id: str
    event_id: str
    event_kind: str
    interval_start_assay_h: float
    interval_end_assay_h: float
    estimate_assay_h: float
    estimate_method: str
    uncertainty_h: float
    post_event_coverage_h: float
    declaration: str


@dataclass(frozen=True)
class ExperimentSource:
    experiment_id: str
    response: pd.DataFrame
    magnitude: pd.DataFrame
    trajectory: pd.DataFrame
    event: EventInterval


def build_experiment_source(
    *,
    experiment_id: str,
    response_frame: pd.DataFrame,
    magnitude_frame: pd.DataFrame,
    trajectory_frame: pd.DataFrame,
    source_spec: ResponseWindowSourceSpec,
    event_spec: EventSpec,
) -> ExperimentSource:
    """Normalize three already-resolved records into one analysis source."""

    state_column = source_spec.state_column
    treatment_map = source_spec.state_values
    case_sensitive = source_spec.state_values_case_sensitive
    if set(treatment_map) != set(STATE_ORDER) or len(set(treatment_map.values())) != len(STATE_ORDER):
        raise ValueError(f"{experiment_id}: resolved state map must define four distinct 00, 10, 01, and 11 values.")
    response = _load_signal(
        response_frame,
        channel=source_spec.response_channel,
        state_column=state_column,
        treatment_map=treatment_map,
        case_sensitive=case_sensitive,
        event_spec=event_spec,
        context=f"{experiment_id}:response",
    )
    magnitude = _load_signal(
        magnitude_frame,
        channel=source_spec.magnitude_channel,
        state_column=state_column,
        treatment_map=treatment_map,
        case_sensitive=case_sensitive,
        event_spec=event_spec,
        context=f"{experiment_id}:magnitude",
    )
    trajectory = _load_signal(
        trajectory_frame,
        channel=source_spec.growth_channel,
        state_column=state_column,
        treatment_map=treatment_map,
        case_sensitive=case_sensitive,
        event_spec=event_spec,
        context=f"{experiment_id}:growth",
        require_positive=False,
    )
    event = resolve_event_interval(response, experiment_id=experiment_id, event_spec=event_spec)
    magnitude_event = resolve_event_interval(magnitude, experiment_id=experiment_id, event_spec=event_spec)
    growth_event = resolve_event_interval(trajectory, experiment_id=experiment_id, event_spec=event_spec)
    _require_event_parity(event, magnitude_event, context=f"{experiment_id}:response/magnitude")
    _require_event_parity(event, growth_event, context=f"{experiment_id}:response/growth")

    for frame in (response, magnitude, trajectory):
        frame["experiment_id"] = experiment_id
        frame["time_from_event_h"] = frame["time"].to_numpy(dtype=float) - event.estimate_assay_h
    reference_id = source_spec.reference_design_id
    if reference_id not in set(magnitude["design_id"].astype(str)):
        raise ValueError(f"{experiment_id}: reference design {reference_id!r} is absent from magnitude records.")

    return ExperimentSource(
        experiment_id=experiment_id,
        response=response,
        magnitude=magnitude,
        trajectory=trajectory,
        event=event,
    )


def resolve_event_interval(
    frame: pd.DataFrame,
    *,
    experiment_id: str,
    event_spec: EventSpec,
) -> EventInterval:
    segments = pd.to_numeric(frame[event_spec.segment_column], errors="coerce")
    segment_values = segments.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(segment_values).all() or not np.equal(segment_values, np.trunc(segment_values)).all():
        raise ValueError(f"{experiment_id}: event segment indexes must be finite integers.")
    segment_indexes = segments.astype(int)
    indexes = set(segment_indexes)
    expected = {event_spec.pre_segment_index, event_spec.post_segment_index}
    if indexes != expected:
        raise ValueError(
            f"{experiment_id}: event requires segment indexes {sorted(expected)}; found {sorted(indexes)}."
        )
    times = pd.to_numeric(frame["time"], errors="coerce")
    if not np.isfinite(times.to_numpy(dtype=float)).all():
        raise ValueError(f"{experiment_id}: event alignment requires finite acquisition times.")
    pre = times.loc[segment_indexes.eq(event_spec.pre_segment_index)]
    post = times.loc[segment_indexes.eq(event_spec.post_segment_index)]
    last_pre = float(pre.max())
    first_post = float(post.min())
    assay_end = float(post.max())
    if not last_pre < first_post <= assay_end:
        raise ValueError(
            f"{experiment_id}: event segments are not chronological: "
            f"last_pre={last_pre}, first_post={first_post}, assay_end={assay_end}."
        )
    estimate = (last_pre + first_post) / 2.0
    return EventInterval(
        experiment_id=experiment_id,
        event_id=event_spec.event_id,
        event_kind=event_spec.event_kind,
        interval_start_assay_h=last_pre,
        interval_end_assay_h=first_post,
        estimate_assay_h=estimate,
        estimate_method=event_spec.estimate_method,
        uncertainty_h=(first_post - last_pre) / 2.0,
        post_event_coverage_h=assay_end - first_post,
        declaration=event_spec.declaration,
    )


def _load_signal(
    frame: pd.DataFrame,
    *,
    channel: str,
    state_column: str,
    treatment_map: Mapping[str, str],
    case_sensitive: bool,
    event_spec: EventSpec,
    context: str,
    require_positive: bool = True,
) -> pd.DataFrame:
    frame = frame.copy()
    required = {"design_id", "position", "time", "channel", "value", state_column, event_spec.segment_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} record is missing columns: {missing}.")
    work = frame.loc[frame["channel"].astype(str).eq(channel)].copy()
    if work.empty:
        raise ValueError(f"{context} record has no rows for channel {channel!r}.")
    reverse = {(value if case_sensitive else value.strip().casefold()): state for state, value in treatment_map.items()}
    values = work[state_column].astype(str)
    if not case_sensitive:
        values = values.str.strip().str.casefold()
    work["state"] = values.map(reverse)
    unknown = sorted(work.loc[work["state"].isna(), state_column].astype(str).unique().tolist())
    if unknown:
        raise ValueError(f"{context} record contains unmapped state values: {unknown}.")
    work["design_id"] = work["design_id"].astype(str)
    work["position"] = work["position"].astype(str)
    work["state"] = work["state"].astype(str)
    work["time"] = pd.to_numeric(work["time"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    if not np.isfinite(work[["time", "value"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{context} record contains non-finite time or values.")
    if require_positive and (work["value"].to_numpy(dtype=float) <= 0.0).any():
        raise ValueError(
            f"{context} record contains non-positive values; "
            "response-window log-space reduction requires strictly positive source values."
        )
    _normalize_value_provenance(work, context=context)
    return work.loc[
        :,
        [
            "design_id",
            "position",
            "state",
            "time",
            "channel",
            "value",
            "value_policy_clipped",
            "value_instrument_overflow",
            "value_bound_kind",
            event_spec.segment_column,
        ],
    ].reset_index(drop=True)


def _normalize_value_provenance(frame: pd.DataFrame, *, context: str) -> None:
    required = {"value_policy_clipped", "value_instrument_overflow", "value_bound_kind"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} record is missing required value provenance columns: {missing}.")
    policy = _boolean_provenance(frame["value_policy_clipped"], context=f"{context}:value_policy_clipped")
    overflow = _boolean_provenance(frame["value_instrument_overflow"], context=f"{context}:value_instrument_overflow")
    frame["value_policy_clipped"] = policy
    frame["value_instrument_overflow"] = overflow
    bounds = frame["value_bound_kind"].astype(str)
    allowed = {"exact", "lower", "upper", "indeterminate"}
    unknown = sorted(set(bounds) - allowed)
    if unknown:
        raise ValueError(f"{context} record contains unsupported value_bound_kind values: {unknown}.")
    if ((policy | overflow) & bounds.eq("exact")).any():
        raise ValueError(f"{context} record marks clipped or overflowed values as exact.")
    if (bounds.ne("exact") & ~(policy | overflow)).any():
        raise ValueError(f"{context} record contains a value bound without clipping or overflow provenance.")
    frame["value_bound_kind"] = bounds


def _boolean_provenance(values: pd.Series, *, context: str) -> pd.Series:
    if values.isna().any() or not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{context} must contain booleans without missing values.")
    return values.astype(bool)


def _require_event_parity(left: EventInterval, right: EventInterval, *, context: str) -> None:
    fields = (
        "interval_start_assay_h",
        "interval_end_assay_h",
        "estimate_assay_h",
        "post_event_coverage_h",
    )
    if any(not np.isclose(getattr(left, field), getattr(right, field), rtol=0.0, atol=1.0e-12) for field in fields):
        raise ValueError(f"{context} event bounds disagree across source records.")


__all__ = [
    "ANNOTATED_CONTRACT",
    "STATE_ORDER",
    "EventInterval",
    "ExperimentSource",
    "build_experiment_source",
    "resolve_event_interval",
]
