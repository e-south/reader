"""Materialize well and design four-state event-window records from verified sources."""

from __future__ import annotations

import pandas as pd

from .aggregation import build_design_records
from .contracts import FourStateEventWindowAnalysisSpec, ReductionSpec
from .disposition_records import disposition_frame, partition_reduction_wells
from .observation_resampling import descriptive_resampling_records
from .reduction import (
    combine_bound_kinds,
    four_state_event_window_temporal_spec,
    invert_bound_kind,
    reduce_temporal_trace,
)
from .sources import ExperimentSource, event_record
from .well_exclusion_validation import require_well_exclusion_support_failure


def materialize_experiment(
    source: ExperimentSource,
    *,
    request: FourStateEventWindowAnalysisSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build midpoint well/design records plus event-bound sensitivity."""

    require_well_exclusion_support_failure(
        source.well_exclusions,
        reductions={item.id: item for item in request.reductions},
        quality=request.quality,
        response=source.response,
        magnitude=source.magnitude,
        event_estimates_h=(
            source.event.interval_start_assay_h,
            source.event.estimate_assay_h,
            source.event.interval_end_assay_h,
        ),
        experiment_id=source.experiment_id,
    )
    well_frames: list[pd.DataFrame] = []
    design_frames: list[pd.DataFrame] = []
    draw_frames: list[pd.DataFrame] = []
    disposition_records: list[dict[str, object]] = []
    for reduction in request.reductions:
        midpoint = _reduce_wells(
            source,
            request=request,
            reduction=reduction,
            event_estimate_h=source.event.estimate_assay_h,
        )
        lower = _reduce_wells(
            source,
            request=request,
            reduction=reduction,
            event_estimate_h=source.event.interval_start_assay_h,
        )
        upper = _reduce_wells(
            source,
            request=request,
            reduction=reduction,
            event_estimate_h=source.event.interval_end_assay_h,
        )
        eligible_midpoint, eligible_lower, eligible_upper, reduction_dispositions = partition_reduction_wells(
            midpoint,
            lower,
            upper,
            exclusions=request.source.well_exclusions,
            dispositions=request.source.design_dispositions,
            experiment_id=source.experiment_id,
            reduction_id=reduction.id,
            reference_design_id=request.source.reference_design_id,
            available_design_ids=set(source.response["design_id"].astype(str)),
            required_count=request.quality.min_observations_per_state,
        )
        well_frames.append(midpoint)
        draw_frames.append(descriptive_resampling_records(eligible_midpoint, request=request))
        design_frames.append(
            build_design_records(
                eligible_midpoint,
                lower=eligible_lower,
                upper=eligible_upper,
                source=source,
                request=request,
                reduction=reduction,
            )
        )
        disposition_records.extend(reduction_dispositions)

    wells = pd.concat(well_frames, ignore_index=True)
    designs = pd.concat(design_frames, ignore_index=True)
    draws = pd.concat(draw_frames, ignore_index=True)
    well_key = ["experiment_id", "design_id", "state", "position", "reduction_id"]
    design_key = ["experiment_id", "design_id", "reduction_id"]
    if wells.duplicated(subset=well_key).any():
        raise ValueError(f"{source.experiment_id}: well record identity is not unique.")
    if designs.duplicated(subset=design_key).any():
        raise ValueError(f"{source.experiment_id}: design record identity is not unique.")
    draw_key = ["experiment_id", "design_id", "reduction_id", "draw_index"]
    if draws.duplicated(subset=draw_key).any():
        raise ValueError(f"{source.experiment_id}: descriptive-resampling draw identity is not unique.")
    traces = _trace_record(source, request=request)
    return wells, designs, draws, traces, event_record(source.event), disposition_frame(disposition_records)


def _reduce_wells(
    source: ExperimentSource,
    *,
    request: FourStateEventWindowAnalysisSpec,
    reduction: ReductionSpec,
    event_estimate_h: float,
) -> pd.DataFrame:
    response = _summaries_by_trace(
        source.response,
        request=request,
        reduction=reduction,
        event_estimate_h=event_estimate_h,
        signal_kind="response",
        pre_window_end_h=source.event.interval_start_assay_h,
    )
    magnitude = _summaries_by_trace(
        source.magnitude,
        request=request,
        reduction=reduction,
        event_estimate_h=event_estimate_h,
        signal_kind="magnitude",
        pre_window_end_h=None,
    )
    keys = ["design_id", "state", "position"]
    merged = response.merge(
        magnitude, on=keys, how="outer", validate="one_to_one", suffixes=("_response", "_magnitude")
    )
    if merged.isna().any().any():
        missing = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"{source.experiment_id}:{reduction.id}: response/magnitude wells do not align: {missing}.")
    merged.insert(0, "experiment_id", source.experiment_id)
    merged["reduction_id"] = reduction.id
    merged["reduction_method"] = reduction.method
    merged["response_basis"] = reduction.response_basis
    merged["reduction_role"] = reduction.role
    merged["event_time_estimate_assay_h"] = float(event_estimate_h)
    merged["window_start_event_h"] = reduction.window_start_event_h
    merged["window_end_event_h"] = reduction.window_end_event_h
    merged["window_start_assay_h"] = event_estimate_h + reduction.window_start_event_h
    merged["window_end_assay_h"] = event_estimate_h + reduction.window_end_event_h
    merged["is_reference"] = merged["design_id"].astype(str).eq(request.source.reference_design_id)
    return merged.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _summaries_by_trace(
    frame: pd.DataFrame,
    *,
    request: FourStateEventWindowAnalysisSpec,
    reduction: ReductionSpec,
    event_estimate_h: float,
    signal_kind: str,
    pre_window_end_h: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    excluded = {
        (item.design_id, item.state, item.position)
        for item in request.source.well_exclusions
        if item.experiment_id == source_experiment_id(frame) and item.reduction_id == reduction.id
    }
    for (design_id, state, position), trace in frame.groupby(
        ["design_id", "state", "position"], sort=True, dropna=False
    ):
        if (str(design_id), str(state), str(position)) in excluded:
            continue
        trace_id = f"{frame['experiment_id'].iloc[0]}:{signal_kind}:{design_id}:{state}:{position}:{reduction.id}"
        post = reduce_temporal_trace(
            trace["time"].to_numpy(dtype=float),
            trace["value"].to_numpy(dtype=float),
            spec=four_state_event_window_temporal_spec(reduction, request.quality),
            origin_h=event_estimate_h,
            trace_id=trace_id,
            policy_clipped=trace["value_policy_clipped"].to_numpy(dtype=bool),
            instrument_overflow=trace["value_instrument_overflow"].to_numpy(dtype=bool),
            bound_kinds=trace["value_bound_kind"].to_numpy(dtype=object),
        )
        value = post.value
        policy_clipped_point_count = post.policy_clipped_point_count
        instrument_overflow_point_count = post.instrument_overflow_point_count
        bound_kind = post.bound_kind
        pre_observed_point_count = 0
        pre_integration_point_count = 0
        pre_max_interior_gap_h = 0.0
        if signal_kind == "response" and reduction.response_basis == "post_minus_pre":
            if reduction.pre_window_duration_h is None or pre_window_end_h is None:
                raise ValueError(f"{trace_id}: delta response lacks an explicit pre-event window.")
            pre = reduce_temporal_trace(
                trace["time"].to_numpy(dtype=float),
                trace["value"].to_numpy(dtype=float),
                spec=four_state_event_window_temporal_spec(
                    reduction,
                    request.quality,
                    absolute_window_h=(
                        pre_window_end_h - reduction.pre_window_duration_h,
                        pre_window_end_h,
                    ),
                ),
                trace_id=f"{trace_id}:pre",
                policy_clipped=trace["value_policy_clipped"].to_numpy(dtype=bool),
                instrument_overflow=trace["value_instrument_overflow"].to_numpy(dtype=bool),
                bound_kinds=trace["value_bound_kind"].to_numpy(dtype=object),
            )
            value -= pre.value
            policy_clipped_point_count += pre.policy_clipped_point_count
            instrument_overflow_point_count += pre.instrument_overflow_point_count
            bound_kind = combine_bound_kinds(bound_kind, invert_bound_kind(pre.bound_kind))
            pre_observed_point_count = pre.observed_point_count
            pre_integration_point_count = pre.evaluation_point_count
            pre_max_interior_gap_h = pre.max_interior_gap_h
        rows.append(
            {
                "design_id": str(design_id),
                "state": str(state),
                "position": str(position),
                f"{signal_kind}_well": float(value),
                f"{signal_kind}_observed_point_count": post.observed_point_count,
                f"{signal_kind}_integration_point_count": post.evaluation_point_count,
                f"{signal_kind}_max_interior_gap_h": post.max_interior_gap_h,
                f"{signal_kind}_pre_observed_point_count": pre_observed_point_count,
                f"{signal_kind}_pre_integration_point_count": pre_integration_point_count,
                f"{signal_kind}_pre_max_interior_gap_h": pre_max_interior_gap_h,
                f"{signal_kind}_policy_clipped_point_count": policy_clipped_point_count,
                f"{signal_kind}_instrument_overflow_point_count": instrument_overflow_point_count,
                f"{signal_kind}_bound_kind": bound_kind,
            }
        )
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        raise ValueError(f"{signal_kind} reduction produced no well records.")
    return result


def source_experiment_id(frame: pd.DataFrame) -> str:
    return str(frame["experiment_id"].iloc[0])


def _trace_record(source: ExperimentSource, *, request: FourStateEventWindowAnalysisSpec) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for signal_kind, frame in (
        ("response", source.response),
        ("magnitude", source.magnitude),
        ("growth", source.trajectory),
    ):
        selected = frame.loc[
            :,
            [
                "experiment_id",
                "design_id",
                "position",
                "state",
                "time",
                "time_from_event_h",
                "value",
                "value_policy_clipped",
                "value_instrument_overflow",
                "value_bound_kind",
            ],
        ].copy()
        selected["signal_kind"] = signal_kind
        selected["is_reference"] = selected["design_id"].astype(str).eq(request.source.reference_design_id)
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    key = ["experiment_id", "design_id", "position", "state", "signal_kind", "time"]
    if result.duplicated(subset=key).any():
        raise ValueError(f"{source.experiment_id}: trace record identity is not unique.")
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


__all__ = ["materialize_experiment"]
