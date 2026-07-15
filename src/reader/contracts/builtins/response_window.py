"""Exact public contracts for plate-reader response-window bundles."""

from __future__ import annotations

from ..model import ColumnRule, DataFrameContract

_STATES = ["00", "10", "01", "11"]
_REDUCTION_ROLES = ["primary", "sensitivity"]
_REDUCTION_METHODS = ["geometric_time_mean", "integrated_linear_mean"]
_RESPONSE_BASES = ["post_window", "post_minus_pre"]
_VALUE_BOUND_KINDS = ["exact", "lower", "upper", "indeterminate"]


def _state_summary_columns() -> list[ColumnRule]:
    columns: list[ColumnRule] = []
    for state in _STATES:
        columns.extend(
            [
                ColumnRule(f"r{state}", "float"),
                ColumnRule(f"b{state}", "float"),
                ColumnRule(f"r{state}_bootstrap_sd", "float", nonnegative=True),
                ColumnRule(f"b{state}_bootstrap_sd", "float", nonnegative=True),
                ColumnRule(f"r{state}_ci_low", "float"),
                ColumnRule(f"r{state}_ci_high", "float"),
                ColumnRule(f"b{state}_ci_low", "float"),
                ColumnRule(f"b{state}_ci_high", "float"),
                ColumnRule(f"r{state}_event_half_range", "float", nonnegative=True),
                ColumnRule(f"b{state}_event_half_range", "float", nonnegative=True),
                ColumnRule(f"r{state}_event_sensitivity_has_policy_clipping", "bool"),
                ColumnRule(f"r{state}_event_sensitivity_has_instrument_overflow", "bool"),
                ColumnRule(f"b{state}_event_sensitivity_has_policy_clipping", "bool"),
                ColumnRule(f"b{state}_event_sensitivity_has_instrument_overflow", "bool"),
                ColumnRule(f"r{state}_has_policy_clipping", "bool"),
                ColumnRule(f"r{state}_has_instrument_overflow", "bool"),
                ColumnRule(f"r{state}_bound_kind", "string", allowed_values=_VALUE_BOUND_KINDS),
                ColumnRule(f"b{state}_has_policy_clipping", "bool"),
                ColumnRule(f"b{state}_has_instrument_overflow", "bool"),
                ColumnRule(f"b{state}_bound_kind", "string", allowed_values=_VALUE_BOUND_KINDS),
                ColumnRule(f"n{state}", "int", nonnegative=True),
            ]
        )
    return columns


CONTRACTS: tuple[DataFrameContract, ...] = (
    DataFrameContract(
        id="plate_reader.response_window.wells.v3",
        description="Event-relative well summary with clipping and censor-bound support.",
        columns=[
            ColumnRule("experiment_id", "string"),
            ColumnRule("design_id", "string"),
            ColumnRule("state", "string", allowed_values=_STATES),
            ColumnRule("position", "string"),
            ColumnRule("response_well", "float"),
            ColumnRule("response_observed_point_count", "int", nonnegative=True),
            ColumnRule("response_integration_point_count", "int", nonnegative=True),
            ColumnRule("response_max_interior_gap_h", "float", nonnegative=True),
            ColumnRule("response_pre_observed_point_count", "int", nonnegative=True),
            ColumnRule("response_pre_integration_point_count", "int", nonnegative=True),
            ColumnRule("response_pre_max_interior_gap_h", "float", nonnegative=True),
            ColumnRule("response_policy_clipped_point_count", "int", nonnegative=True),
            ColumnRule("response_instrument_overflow_point_count", "int", nonnegative=True),
            ColumnRule("response_bound_kind", "string", allowed_values=_VALUE_BOUND_KINDS),
            ColumnRule("magnitude_well", "float"),
            ColumnRule("magnitude_observed_point_count", "int", nonnegative=True),
            ColumnRule("magnitude_integration_point_count", "int", nonnegative=True),
            ColumnRule("magnitude_max_interior_gap_h", "float", nonnegative=True),
            ColumnRule("magnitude_pre_observed_point_count", "int", nonnegative=True),
            ColumnRule("magnitude_pre_integration_point_count", "int", nonnegative=True),
            ColumnRule("magnitude_pre_max_interior_gap_h", "float", nonnegative=True),
            ColumnRule("magnitude_policy_clipped_point_count", "int", nonnegative=True),
            ColumnRule("magnitude_instrument_overflow_point_count", "int", nonnegative=True),
            ColumnRule("magnitude_bound_kind", "string", allowed_values=_VALUE_BOUND_KINDS),
            ColumnRule("reduction_id", "string"),
            ColumnRule("reduction_method", "string", allowed_values=_REDUCTION_METHODS),
            ColumnRule("response_basis", "string", allowed_values=_RESPONSE_BASES),
            ColumnRule("reduction_role", "string", allowed_values=_REDUCTION_ROLES),
            ColumnRule("event_time_estimate_assay_h", "float", nonnegative=True),
            ColumnRule("window_start_event_h", "float", nonnegative=True),
            ColumnRule("window_end_event_h", "float", nonnegative=True),
            ColumnRule("window_start_assay_h", "float", nonnegative=True),
            ColumnRule("window_end_assay_h", "float", nonnegative=True),
            ColumnRule("is_reference", "bool"),
        ],
        unique_keys=[["experiment_id", "design_id", "state", "position", "reduction_id"]],
        domain="plate_reader",
        kind="response-window-well-summary",
        allow_extra_columns=False,
    ),
    DataFrameContract(
        id="plate_reader.response_window.designs.v3",
        description="Four-condition response and fluorescence summary with state-level censor bounds.",
        columns=[
            ColumnRule("experiment_id", "string"),
            ColumnRule("design_id", "string"),
            ColumnRule("reference_design_id", "string"),
            ColumnRule("reduction_id", "string"),
            ColumnRule("reduction_method", "string", allowed_values=_REDUCTION_METHODS),
            ColumnRule("response_basis", "string", allowed_values=_RESPONSE_BASES),
            ColumnRule("reduction_role", "string", allowed_values=_REDUCTION_ROLES),
            ColumnRule("replicate_stat", "string", allowed_values=["mean", "median"]),
            ColumnRule("bootstrap_samples", "int", nonnegative=True),
            ColumnRule("confidence_level", "float", nonnegative=True),
            ColumnRule("event_id", "string"),
            ColumnRule("event_time_estimate_assay_h", "float", nonnegative=True),
            ColumnRule("event_time_uncertainty_h", "float", nonnegative=True),
            ColumnRule("window_start_event_h", "float", nonnegative=True),
            ColumnRule("window_end_event_h", "float", nonnegative=True),
            ColumnRule("is_reference", "bool"),
            ColumnRule("min_replicates_per_state", "int", nonnegative=True),
            ColumnRule("min_observed_points_per_trace", "int", nonnegative=True),
            ColumnRule("max_interior_gap_h", "float", nonnegative=True),
            ColumnRule("min_pre_observed_points_per_trace", "int", nonnegative=True),
            ColumnRule("max_pre_interior_gap_h", "float", nonnegative=True),
            *_state_summary_columns(),
        ],
        unique_keys=[["experiment_id", "design_id", "reduction_id"]],
        domain="plate_reader",
        kind="response-window-design-summary",
        allow_extra_columns=False,
    ),
    DataFrameContract(
        id="plate_reader.response_window.bootstrap_draws.v2",
        description="Joint replicate-bootstrap draw over eight response-window components.",
        columns=[
            ColumnRule("experiment_id", "string"),
            ColumnRule("design_id", "string"),
            ColumnRule("reduction_id", "string"),
            ColumnRule("draw_index", "int", nonnegative=True),
            *(ColumnRule(column, "float") for state in _STATES for column in (f"r{state}", f"b{state}")),
            ColumnRule("is_reference", "bool"),
        ],
        unique_keys=[["experiment_id", "design_id", "reduction_id", "draw_index"]],
        domain="plate_reader",
        kind="response-window-bootstrap-draw",
        allow_extra_columns=False,
    ),
    DataFrameContract(
        id="plate_reader.response_window.traces.v3",
        description="Source trace observations with clipping and censor bounds aligned to one event.",
        columns=[
            ColumnRule("experiment_id", "string"),
            ColumnRule("design_id", "string"),
            ColumnRule("position", "string"),
            ColumnRule("state", "string", allowed_values=_STATES),
            ColumnRule("time", "float", nonnegative=True),
            ColumnRule("time_from_event_h", "float"),
            ColumnRule("value", "float"),
            ColumnRule("value_policy_clipped", "bool"),
            ColumnRule("value_instrument_overflow", "bool"),
            ColumnRule("value_bound_kind", "string", allowed_values=_VALUE_BOUND_KINDS),
            ColumnRule("signal_kind", "string", allowed_values=["response", "magnitude", "growth"]),
            ColumnRule("is_reference", "bool"),
        ],
        unique_keys=[["experiment_id", "design_id", "position", "state", "signal_kind", "time"]],
        domain="plate_reader",
        kind="response-window-trace",
        allow_extra_columns=False,
    ),
    DataFrameContract(
        id="plate_reader.response_window.events.v2",
        description="Declared intervention interval and event-relative coverage for one experiment.",
        columns=[
            ColumnRule("experiment_id", "string"),
            ColumnRule("event_id", "string"),
            ColumnRule("event_kind", "string"),
            ColumnRule("event_interval_start_assay_h", "float", nonnegative=True),
            ColumnRule("event_interval_end_assay_h", "float", nonnegative=True),
            ColumnRule("event_time_estimate_assay_h", "float", nonnegative=True),
            ColumnRule("event_time_estimate_method", "string", allowed_values=["segment_gap_midpoint"]),
            ColumnRule("event_time_uncertainty_h", "float", nonnegative=True),
            ColumnRule("post_event_coverage_h", "float", nonnegative=True),
            ColumnRule("declaration", "string"),
        ],
        unique_keys=[["experiment_id"]],
        domain="plate_reader",
        kind="response-window-event",
        allow_extra_columns=False,
    ),
)

__all__ = ["CONTRACTS"]
