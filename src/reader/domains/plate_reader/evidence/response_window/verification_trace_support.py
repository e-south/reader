"""Verify censor provenance against the trace observations that support it."""

from __future__ import annotations

import pandas as pd

from reader.domains.plate_reader.analysis.response_window.contracts import ResponseWindowRequest
from reader.domains.plate_reader.analysis.response_window.reduction import (
    combine_bound_kinds,
    invert_bound_kind,
    summarize_value_provenance,
)
from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER


def verify_trace_support(
    designs: pd.DataFrame,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    *,
    events: pd.DataFrame,
    request: ResponseWindowRequest,
) -> None:
    """Verify midpoint-well and event-sensitivity censor provenance."""

    _verify_well_support(wells, traces, events=events, request=request)
    _verify_event_sensitivity(designs, traces, events=events, request=request)


def _verify_well_support(
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    *,
    events: pd.DataFrame,
    request: ResponseWindowRequest,
) -> None:
    trace_groups = {
        tuple(map(str, key)): frame
        for key, frame in traces.groupby(
            ["experiment_id", "design_id", "state", "position", "signal_kind"],
            sort=False,
        )
    }
    reductions = {reduction.id: reduction for reduction in request.reductions}
    pre_window_ends = events.set_index(events["experiment_id"].astype(str))["event_interval_start_assay_h"].to_dict()
    for row in wells.itertuples(index=False):
        reduction = reductions[str(row.reduction_id)]
        for signal in ("response", "magnitude"):
            key = tuple(map(str, (row.experiment_id, row.design_id, row.state, row.position, signal)))
            try:
                trace = trace_groups[key]
            except KeyError as exc:
                raise ValueError(f"response-window well lacks {signal} trace support: {key!r}.") from exc
            expected = _window_summary(
                trace,
                start=float(row.window_start_assay_h),
                end=float(row.window_end_assay_h),
                trace_id=":".join(key),
            )
            policy_count = expected.policy_clipped_point_count
            overflow_count = expected.instrument_overflow_point_count
            bound_kind = expected.bound_kind
            if signal == "response" and reduction.response_basis == "post_minus_pre":
                if reduction.pre_window_duration_h is None:
                    raise ValueError(f"response-window reduction {reduction.id!r} lacks its pre-window duration.")
                pre_end = float(pre_window_ends[str(row.experiment_id)])
                pre = _window_summary(
                    trace,
                    start=pre_end - reduction.pre_window_duration_h,
                    end=pre_end,
                    trace_id=f"{':'.join(key)}:pre",
                )
                policy_count += pre.policy_clipped_point_count
                overflow_count += pre.instrument_overflow_point_count
                bound_kind = combine_bound_kinds(bound_kind, invert_bound_kind(pre.bound_kind))
            observed = (
                int(getattr(row, f"{signal}_policy_clipped_point_count")),
                int(getattr(row, f"{signal}_instrument_overflow_point_count")),
                str(getattr(row, f"{signal}_bound_kind")),
            )
            if observed != (policy_count, overflow_count, bound_kind):
                raise ValueError("response-window well provenance disagrees with trace support.")


def _verify_event_sensitivity(
    designs: pd.DataFrame,
    traces: pd.DataFrame,
    *,
    events: pd.DataFrame,
    request: ResponseWindowRequest,
) -> None:
    grouped: dict[tuple[str, str, str, str], list[pd.DataFrame]] = {}
    for key, frame in traces.groupby(
        ["experiment_id", "design_id", "state", "signal_kind", "position"],
        sort=False,
    ):
        experiment_id, design_id, state, signal_kind, _position = map(str, key)
        grouped.setdefault((experiment_id, design_id, state, signal_kind), []).append(frame)
    event_rows = {str(row.experiment_id): row for row in events.itertuples(index=False)}
    reductions = {reduction.id: reduction for reduction in request.reductions}
    for row in designs.itertuples(index=False):
        experiment_id = str(row.experiment_id)
        event = event_rows[experiment_id]
        reduction = reductions[str(row.reduction_id)]
        estimates = (
            float(event.event_time_estimate_assay_h),
            float(event.event_interval_start_assay_h),
            float(event.event_interval_end_assay_h),
        )
        for state in STATE_ORDER:
            for prefix, signal_kind in (("r", "response"), ("b", "magnitude")):
                source_design_ids = {str(row.design_id)}
                if signal_kind == "magnitude":
                    source_design_ids.add(str(row.reference_design_id))
                expected = _event_censoring(
                    grouped,
                    source_design_ids=source_design_ids,
                    experiment_id=experiment_id,
                    state=state,
                    signal_kind=signal_kind,
                    estimates=estimates,
                    reduction=reduction,
                    pre_window_end=float(event.event_interval_start_assay_h),
                )
                observed = (
                    bool(getattr(row, f"{prefix}{state}_event_sensitivity_has_policy_clipping")),
                    bool(getattr(row, f"{prefix}{state}_event_sensitivity_has_instrument_overflow")),
                )
                if observed != expected:
                    raise ValueError("response-window event-sensitivity provenance disagrees with trace support.")


def _event_censoring(
    grouped: dict[tuple[str, str, str, str], list[pd.DataFrame]],
    *,
    source_design_ids: set[str],
    experiment_id: str,
    state: str,
    signal_kind: str,
    estimates: tuple[float, float, float],
    reduction,
    pre_window_end: float,
) -> tuple[bool, bool]:
    policy_clipped = False
    instrument_overflow = False
    for design_id in source_design_ids:
        key = (experiment_id, design_id, state, signal_kind)
        try:
            source_traces = grouped[key]
        except KeyError as exc:
            raise ValueError(f"response-window design lacks event-sensitivity trace support: {key!r}.") from exc
        for trace in source_traces:
            summaries = [
                _window_summary(
                    trace,
                    start=estimate + reduction.window_start_event_h,
                    end=estimate + reduction.window_end_event_h,
                    trace_id=":".join(key),
                )
                for estimate in estimates
            ]
            if signal_kind == "response" and reduction.response_basis == "post_minus_pre":
                if reduction.pre_window_duration_h is None:
                    raise ValueError(f"response-window reduction {reduction.id!r} lacks its pre-window duration.")
                summaries.append(
                    _window_summary(
                        trace,
                        start=pre_window_end - reduction.pre_window_duration_h,
                        end=pre_window_end,
                        trace_id=f"{':'.join(key)}:pre",
                    )
                )
            policy_clipped |= any(summary.policy_clipped_point_count > 0 for summary in summaries)
            instrument_overflow |= any(summary.instrument_overflow_point_count > 0 for summary in summaries)
    return policy_clipped, instrument_overflow


def _window_summary(trace: pd.DataFrame, *, start: float, end: float, trace_id: str):
    return summarize_value_provenance(
        trace["time"].to_numpy(dtype=float),
        window_start_h=start,
        window_end_h=end,
        policy_clipped=trace["value_policy_clipped"].to_numpy(dtype=bool),
        instrument_overflow=trace["value_instrument_overflow"].to_numpy(dtype=bool),
        bound_kinds=trace["value_bound_kind"].to_numpy(dtype=object),
        trace_id=trace_id,
    )


__all__ = ["verify_trace_support"]
