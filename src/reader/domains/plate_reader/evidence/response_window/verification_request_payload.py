"""Persisted response-window payload checks against its bundled request."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.contracts import ResponseWindowRequest
from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER


def verify_request_payload_parity(
    request: ResponseWindowRequest,
    *,
    frames: dict[str, pd.DataFrame],
    source_records: tuple[dict[str, object], ...],
) -> None:
    """Reject request claims that disagree with their persisted representation."""

    _verify_experiments(request, events=frames["events"], source_records=source_records)
    _verify_source_record_ids(request, source_records=source_records)
    _verify_event(request, designs=frames["designs"], wells=frames["wells"], events=frames["events"])
    _verify_reductions(request, designs=frames["designs"], wells=frames["wells"])
    _verify_aggregation(request, designs=frames["designs"])
    _verify_quality(request, designs=frames["designs"], wells=frames["wells"], traces=frames["traces"])


def _verify_experiments(
    request: ResponseWindowRequest,
    *,
    events: pd.DataFrame,
    source_records: tuple[dict[str, object], ...],
) -> None:
    expected = set(request.experiment_ids)
    event_ids = set(events["experiment_id"].astype(str))
    source_ids = {str(record["experiment_id"]) for record in source_records}
    if event_ids != expected or source_ids != expected:
        raise ValueError("response-window request experiment identities disagree with persisted payloads.")


def _verify_source_record_ids(
    request: ResponseWindowRequest,
    *,
    source_records: tuple[dict[str, object], ...],
) -> None:
    expected = {
        request.source.response_record_id,
        request.source.magnitude_record_id,
        request.source.trajectory_record_id,
    }
    if any(set(record["records"]) != expected for record in source_records):
        raise ValueError("response-window request source record identities disagree with persisted provenance.")


def _verify_event(
    request: ResponseWindowRequest,
    *,
    designs: pd.DataFrame,
    wells: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    event = request.event
    expected = {
        "event_id": event.event_id,
        "event_kind": event.event_kind,
        "event_time_estimate_method": event.estimate_method,
        "declaration": event.declaration,
    }
    for field, value in expected.items():
        if set(events[field].astype(str)) != {value}:
            label = "identity" if field == "event_id" else field.removeprefix("event_").replace("_", " ")
            raise ValueError(f"response-window request event {label} disagrees with persisted events.")
    if set(designs["event_id"].astype(str)) != {event.event_id}:
        raise ValueError("response-window request event identity disagrees with persisted design rows.")
    event_times = events.set_index("experiment_id")
    for frame, label in ((designs, "design"), (wells, "well")):
        estimates = frame["experiment_id"].map(event_times["event_time_estimate_assay_h"])
        if not np.allclose(frame["event_time_estimate_assay_h"], estimates, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"response-window {label} event estimates disagree with persisted events.")
    uncertainties = designs["experiment_id"].map(event_times["event_time_uncertainty_h"])
    if not np.allclose(designs["event_time_uncertainty_h"], uncertainties, rtol=0.0, atol=1.0e-12):
        raise ValueError("response-window design event uncertainty disagrees with persisted events.")


def _verify_reductions(
    request: ResponseWindowRequest,
    *,
    designs: pd.DataFrame,
    wells: pd.DataFrame,
) -> None:
    columns = {
        "reduction_method": "method",
        "response_basis": "response_basis",
        "reduction_role": "role",
        "window_start_event_h": "window_start_event_h",
        "window_end_event_h": "window_end_event_h",
    }
    expected = {
        spec.id: tuple(getattr(spec, attribute) for attribute in columns.values()) for spec in request.reductions
    }
    for frame in (designs, wells):
        semantics = frame.loc[:, ["reduction_id", *columns]].drop_duplicates()
        if semantics["reduction_id"].astype(str).duplicated().any():
            raise ValueError("response-window persisted reduction identities have conflicting semantics.")
        observed = {
            str(row.reduction_id): tuple(getattr(row, column) for column in columns)
            for row in semantics.itertuples(index=False)
        }
        if observed != expected:
            raise ValueError("response-window request reduction semantics disagree with persisted rows.")
    event_times = wells.groupby("experiment_id")["event_time_estimate_assay_h"].first()
    starts = wells["experiment_id"].map(event_times) + wells["window_start_event_h"]
    ends = wells["experiment_id"].map(event_times) + wells["window_end_event_h"]
    if not np.allclose(wells["window_start_assay_h"], starts, rtol=0.0, atol=1.0e-12) or not np.allclose(
        wells["window_end_assay_h"], ends, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("response-window assay windows disagree with request-relative windows.")


def _verify_aggregation(request: ResponseWindowRequest, *, designs: pd.DataFrame) -> None:
    expected = {
        "replicate_stat": request.aggregation.replicate_stat,
        "bootstrap_samples": request.aggregation.bootstrap_samples,
        "confidence_level": request.aggregation.confidence_level,
    }
    if any(set(designs[field]) != {value} for field, value in expected.items()):
        raise ValueError("response-window request aggregation semantics disagree with persisted design rows.")


def _verify_quality(
    request: ResponseWindowRequest,
    *,
    designs: pd.DataFrame,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
) -> None:
    ratios = traces.loc[traces["signal_kind"].isin(["response", "magnitude"]), "value"].to_numpy(dtype=float)
    if np.any(ratios <= request.quality.positive_floor):
        raise ValueError("response-window persisted ratios violate the request positive floor.")
    gap_columns = [column for column in wells if column.endswith("max_interior_gap_h")]
    design_gap_columns = [column for column in designs if column.endswith("max_interior_gap_h")]
    if any(
        (frame[column] > request.quality.max_interior_gap_h).any()
        for frame, columns in ((wells, gap_columns), (designs, design_gap_columns))
        for column in columns
    ):
        raise ValueError("response-window persisted traces violate the request interior-gap limit.")
    replicate_columns = [f"n{state}" for state in STATE_ORDER]
    if (designs[replicate_columns] < request.quality.min_replicates_per_state).any().any():
        raise ValueError("response-window persisted rows violate the request replicate minimum.")


__all__ = ["verify_request_payload_parity"]
