"""Aggregate response-window well summaries into design-level records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ReductionSpec, ResponseWindowAnalysisSpec
from .provenance import stable_seed
from .reduction import combine_bound_kinds, invert_bound_kind
from .sources import STATE_ORDER, ExperimentSource
from .uncertainty import joint_state_bootstrap_draws, symmetric_event_sensitivity_half_width


def build_design_records(
    midpoint: pd.DataFrame,
    *,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    source: ExperimentSource,
    request: ResponseWindowAnalysisSpec,
    reduction: ReductionSpec,
) -> pd.DataFrame:
    """Aggregate one reduction and its event-bound sensitivity records."""

    midpoint_values = _aggregate_state_values(midpoint, request=request)
    lower_values = _aggregate_state_values(lower, request=request)
    upper_values = _aggregate_state_values(upper, request=request)
    records: list[dict[str, object]] = []
    for design_id, design in midpoint_values.groupby("design_id", sort=True):
        if set(design["state"].astype(str)) != set(STATE_ORDER):
            raise ValueError(f"{source.experiment_id}:{design_id}:{reduction.id} lacks complete four-state support.")
        lower_design = lower_values.loc[lower_values["design_id"].astype(str).eq(str(design_id))].set_index("state")
        upper_design = upper_values.loc[upper_values["design_id"].astype(str).eq(str(design_id))].set_index("state")
        design_by_state = design.set_index("state")
        record: dict[str, object] = {
            "experiment_id": source.experiment_id,
            "design_id": str(design_id),
            "reference_design_id": request.source.reference_design_id,
            "reduction_id": reduction.id,
            "reduction_method": reduction.method,
            "response_basis": reduction.response_basis,
            "reduction_role": reduction.role,
            "replicate_stat": request.aggregation.replicate_stat,
            "bootstrap_samples": request.aggregation.bootstrap_samples,
            "confidence_level": request.aggregation.confidence_level,
            "event_id": source.event.event_id,
            "event_time_estimate_assay_h": source.event.estimate_assay_h,
            "event_time_uncertainty_h": source.event.uncertainty_h,
            "window_start_event_h": reduction.window_start_event_h,
            "window_end_event_h": reduction.window_end_event_h,
            "is_reference": str(design_id) == request.source.reference_design_id,
            "min_replicates_per_state": int(design["replicate_count"].min()),
            "min_observed_points_per_trace": int(design["min_observed_points"].min()),
            "max_interior_gap_h": float(design["max_interior_gap_h"].max()),
            "min_pre_observed_points_per_trace": int(design["min_pre_observed_points"].min()),
            "max_pre_interior_gap_h": float(design["max_pre_interior_gap_h"].max()),
        }
        for state in STATE_ORDER:
            row = design_by_state.loc[state]
            lower_row = lower_design.loc[state]
            upper_row = upper_design.loc[state]
            record[f"r{state}"] = float(row["response"])
            record[f"b{state}"] = float(row["anchored_fluorescence"])
            record[f"r{state}_bootstrap_sd"] = float(row["response_bootstrap_sd"])
            record[f"b{state}_bootstrap_sd"] = float(row["anchored_fluorescence_bootstrap_sd"])
            record[f"r{state}_ci_low"] = float(row["response_ci_low"])
            record[f"r{state}_ci_high"] = float(row["response_ci_high"])
            record[f"b{state}_ci_low"] = float(row["anchored_fluorescence_ci_low"])
            record[f"b{state}_ci_high"] = float(row["anchored_fluorescence_ci_high"])
            record[f"r{state}_event_half_range"] = symmetric_event_sensitivity_half_width(
                float(row["response"]),
                float(lower_row["response"]),
                float(upper_row["response"]),
            )
            record[f"b{state}_event_half_range"] = symmetric_event_sensitivity_half_width(
                float(row["anchored_fluorescence"]),
                float(lower_row["anchored_fluorescence"]),
                float(upper_row["anchored_fluorescence"]),
            )
            for prefix, field in (("r", "response"), ("b", "anchored_fluorescence")):
                record[f"{prefix}{state}_event_sensitivity_has_policy_clipping"] = any(
                    bool(candidate[f"{field}_has_policy_clipping"]) for candidate in (row, lower_row, upper_row)
                )
                record[f"{prefix}{state}_event_sensitivity_has_instrument_overflow"] = any(
                    bool(candidate[f"{field}_has_instrument_overflow"]) for candidate in (row, lower_row, upper_row)
                )
            record[f"r{state}_has_policy_clipping"] = bool(row["response_has_policy_clipping"])
            record[f"r{state}_has_instrument_overflow"] = bool(row["response_has_instrument_overflow"])
            record[f"r{state}_bound_kind"] = str(row["response_bound_kind"])
            record[f"b{state}_has_policy_clipping"] = bool(row["anchored_fluorescence_has_policy_clipping"])
            record[f"b{state}_has_instrument_overflow"] = bool(row["anchored_fluorescence_has_instrument_overflow"])
            record[f"b{state}_bound_kind"] = str(row["anchored_fluorescence_bound_kind"])
            record[f"n{state}"] = int(row["replicate_count"])
        records.append(record)
    return pd.DataFrame.from_records(records)


def _aggregate_state_values(wells: pd.DataFrame, *, request: ResponseWindowAnalysisSpec) -> pd.DataFrame:
    stat = request.aggregation.replicate_stat
    aggregate = np.median if stat == "median" else np.mean
    anchor_id = request.source.reference_design_id
    anchor = wells.loc[wells["design_id"].astype(str).eq(anchor_id)]
    if set(anchor["state"].astype(str)) != set(STATE_ORDER):
        raise ValueError(f"reference design {anchor_id!r} lacks complete four-state support.")
    rows: list[dict[str, object]] = []
    alpha = (1.0 - request.aggregation.confidence_level) / 2.0
    for (design_id, state), frame in wells.groupby(["design_id", "state"], sort=True):
        response_values = frame["response_well"].to_numpy(dtype=float)
        magnitude_values = frame["magnitude_well"].to_numpy(dtype=float)
        state_anchor = anchor.loc[anchor["state"].astype(str).eq(str(state))]
        anchor_values = state_anchor["magnitude_well"].to_numpy(dtype=float)
        minimum = request.quality.min_replicates_per_state
        if min(len(response_values), len(magnitude_values), len(anchor_values)) < minimum:
            raise ValueError(f"{design_id}:{state} requires at least {minimum} design and reference replicate wells.")
        rng = np.random.default_rng(
            stable_seed(
                request.aggregation.random_seed,
                str(frame["experiment_id"].iloc[0]),
                str(design_id),
                str(state),
                str(frame["reduction_id"].iloc[0]),
            )
        )
        response_draws, fluorescence_draws = joint_state_bootstrap_draws(
            response_values,
            magnitude_values,
            anchor_values,
            samples=request.aggregation.bootstrap_samples,
            stat=stat,
            rng=rng,
            paired_anchor=str(design_id) == anchor_id,
        )
        response_bound = combine_bound_kinds(*frame["response_bound_kind"])
        design_magnitude_bound = combine_bound_kinds(*frame["magnitude_bound_kind"])
        anchor_magnitude_bound = combine_bound_kinds(*state_anchor["magnitude_bound_kind"])
        anchored_bound = (
            "exact"
            if str(design_id) == anchor_id
            else combine_bound_kinds(
                design_magnitude_bound,
                invert_bound_kind(anchor_magnitude_bound),
            )
        )
        rows.append(
            {
                "design_id": str(design_id),
                "state": str(state),
                "response": float(aggregate(response_values)),
                "anchored_fluorescence": float(aggregate(magnitude_values) - aggregate(anchor_values)),
                "response_bootstrap_sd": float(np.std(response_draws, ddof=1)),
                "anchored_fluorescence_bootstrap_sd": float(np.std(fluorescence_draws, ddof=1)),
                "response_ci_low": float(np.quantile(response_draws, alpha)),
                "response_ci_high": float(np.quantile(response_draws, 1.0 - alpha)),
                "anchored_fluorescence_ci_low": float(np.quantile(fluorescence_draws, alpha)),
                "anchored_fluorescence_ci_high": float(np.quantile(fluorescence_draws, 1.0 - alpha)),
                "replicate_count": int(min(len(response_values), len(magnitude_values))),
                "min_observed_points": int(
                    frame[["response_observed_point_count", "magnitude_observed_point_count"]].min().min()
                ),
                "max_interior_gap_h": float(
                    frame[["response_max_interior_gap_h", "magnitude_max_interior_gap_h"]].max().max()
                ),
                "min_pre_observed_points": int(frame["response_pre_observed_point_count"].min()),
                "max_pre_interior_gap_h": float(frame["response_pre_max_interior_gap_h"].max()),
                "response_has_policy_clipping": bool(frame["response_policy_clipped_point_count"].gt(0).any()),
                "response_has_instrument_overflow": bool(frame["response_instrument_overflow_point_count"].gt(0).any()),
                "response_bound_kind": response_bound,
                "anchored_fluorescence_has_policy_clipping": bool(
                    frame["magnitude_policy_clipped_point_count"].gt(0).any()
                    or state_anchor["magnitude_policy_clipped_point_count"].gt(0).any()
                ),
                "anchored_fluorescence_has_instrument_overflow": bool(
                    frame["magnitude_instrument_overflow_point_count"].gt(0).any()
                    or state_anchor["magnitude_instrument_overflow_point_count"].gt(0).any()
                ),
                "anchored_fluorescence_bound_kind": anchored_bound,
            }
        )
    return pd.DataFrame.from_records(rows)


__all__ = ["build_design_records"]
