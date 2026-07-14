"""Manifest-backed source loading for response-window analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reader.contracts import ContractCatalog

from .contracts import EventSpec, ResponseSourceSpec
from .provenance import sha256_file

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
    experiment_dir: Path
    config_path: Path
    records_path: Path
    response_path: Path
    magnitude_path: Path
    trajectory_path: Path
    response: pd.DataFrame
    magnitude: pd.DataFrame
    trajectory: pd.DataFrame
    event: EventInterval
    record_digests: dict[str, str]


@dataclass(frozen=True)
class ResolvedExperimentSource:
    experiment_id: str
    experiment_dir: Path
    config_path: Path
    records_path: Path
    record_paths: dict[str, Path]
    record_contracts: dict[str, str]
    record_digests: dict[str, str]
    state_column: str
    treatment_map: dict[str, str]
    state_values_case_sensitive: bool


ExperimentSourceLoader = Callable[[str, ResponseSourceSpec, EventSpec], ExperimentSource]


def load_experiment_source(
    resolved: ResolvedExperimentSource,
    *,
    source_spec: ResponseSourceSpec,
    event_spec: EventSpec,
    contracts: ContractCatalog,
) -> ExperimentSource:
    """Load one experiment from records resolved by the Reader runtime."""

    experiment_id = resolved.experiment_id
    record_contracts = (
        (source_spec.response_record_id, ANNOTATED_CONTRACT),
        (source_spec.magnitude_record_id, ANNOTATED_CONTRACT),
        (source_spec.trajectory_record_id, ANNOTATED_CONTRACT),
    )
    record_paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for record_id, expected_contract in record_contracts:
        path = resolved.record_paths.get(record_id)
        if path is None:
            raise ValueError(f"{experiment_id}: required record {record_id!r} is missing.")
        contract_id = resolved.record_contracts.get(record_id)
        if contract_id != expected_contract:
            raise ValueError(
                f"{experiment_id}: record {record_id!r} has contract {contract_id!r}; expected {expected_contract!r}."
            )
        expected_digest = resolved.record_digests.get(record_id, "")
        if not expected_digest:
            raise ValueError(f"{experiment_id}: record {record_id!r} lacks content_digest.")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"{experiment_id}: record {record_id!r} digest mismatch; "
                f"expected {expected_digest}, observed {actual_digest}."
            )
        digests[record_id] = actual_digest
        record_paths[record_id] = path

    state_column = resolved.state_column
    treatment_map = resolved.treatment_map
    case_sensitive = resolved.state_values_case_sensitive
    if set(treatment_map) != set(STATE_ORDER) or len(set(treatment_map.values())) != len(STATE_ORDER):
        raise ValueError(f"{experiment_id}: resolved state map must define four distinct 00, 10, 01, and 11 values.")
    response = _load_signal(
        record_paths[source_spec.response_record_id],
        channel=source_spec.response_channel,
        state_column=state_column,
        treatment_map=treatment_map,
        case_sensitive=case_sensitive,
        event_spec=event_spec,
        context=f"{experiment_id}:response",
    )
    magnitude = _load_signal(
        record_paths[source_spec.magnitude_record_id],
        channel=source_spec.magnitude_channel,
        state_column=state_column,
        treatment_map=treatment_map,
        case_sensitive=case_sensitive,
        event_spec=event_spec,
        context=f"{experiment_id}:magnitude",
    )
    trajectory = _load_signal(
        record_paths[source_spec.trajectory_record_id],
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
        experiment_dir=resolved.experiment_dir,
        config_path=resolved.config_path,
        records_path=resolved.records_path,
        response_path=record_paths[source_spec.response_record_id],
        magnitude_path=record_paths[source_spec.magnitude_record_id],
        trajectory_path=record_paths[source_spec.trajectory_record_id],
        response=response,
        magnitude=magnitude,
        trajectory=trajectory,
        event=event,
        record_digests=digests,
    )


def resolve_event_interval(
    frame: pd.DataFrame,
    *,
    experiment_id: str,
    event_spec: EventSpec,
) -> EventInterval:
    indexes = set(pd.to_numeric(frame[event_spec.segment_column], errors="coerce").dropna().astype(int))
    expected = {event_spec.pre_segment_index, event_spec.post_segment_index}
    if indexes != expected:
        raise ValueError(
            f"{experiment_id}: event requires segment indexes {sorted(expected)}; found {sorted(indexes)}."
        )
    times = pd.to_numeric(frame["time"], errors="coerce")
    if not np.isfinite(times.to_numpy(dtype=float)).all():
        raise ValueError(f"{experiment_id}: event alignment requires finite acquisition times.")
    pre = times.loc[frame[event_spec.segment_column].astype(int).eq(event_spec.pre_segment_index)]
    post = times.loc[frame[event_spec.segment_column].astype(int).eq(event_spec.post_segment_index)]
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
    path: Path,
    *,
    channel: str,
    state_column: str,
    treatment_map: Mapping[str, str],
    case_sensitive: bool,
    event_spec: EventSpec,
    context: str,
    require_positive: bool = True,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
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
        raise ValueError(f"{context} record contains non-positive ratio values.")
    return work.loc[
        :,
        ["design_id", "position", "state", "time", "channel", "value", event_spec.segment_column],
    ].reset_index(drop=True)


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
    "ExperimentSourceLoader",
    "ResolvedExperimentSource",
    "load_experiment_source",
    "resolve_event_interval",
]
