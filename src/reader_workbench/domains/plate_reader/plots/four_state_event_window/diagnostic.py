"""Select one diagnostic from canonical four-state event-window records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import COMPONENT_COLUMNS, STATE_ORDER

BOUND_KINDS = frozenset({"exact", "lower", "upper", "indeterminate"})


@dataclass(frozen=True)
class FourStateEventWindowDiagnostic:
    """One source design and primary reduction ready for rendering."""

    source_experiment_id: str
    design_id: str
    reference_design_id: str
    reduction_id: str
    reduction_method: str
    response_basis: str
    observation_stat: str
    descriptive_resampling_draws: int
    descriptive_interval_mass: float
    event_id: str
    event_time_uncertainty_h: float
    window: tuple[float, float]
    pre_window: tuple[float, float] | None
    component_values: tuple[float, ...]
    component_descriptive_interval_low: tuple[float, ...]
    component_descriptive_interval_high: tuple[float, ...]
    component_event_half_range: tuple[float, ...]
    component_bound_kinds: tuple[str, ...]
    component_has_policy_clipping: tuple[bool, ...]
    component_has_instrument_overflow: tuple[bool, ...]
    component_event_has_policy_clipping: tuple[bool, ...]
    component_event_has_instrument_overflow: tuple[bool, ...]
    traces: pd.DataFrame


def prepare_four_state_event_window_diagnostic(
    traces: pd.DataFrame,
    designs: pd.DataFrame,
    *,
    source_experiment_id: str,
    design_id: str,
    reduction_id: str,
    pre_window_duration_h: float | None,
) -> FourStateEventWindowDiagnostic:
    """Select and validate one diagnostic subject from canonical records."""

    source_experiment_id = _identifier(source_experiment_id, field="source_experiment_id")
    design_id = _identifier(design_id, field="design_id")
    reduction_id = _identifier(reduction_id, field="reduction_id")
    _require_columns(designs, _design_columns(), record="designs")
    selected = designs.loc[
        designs["experiment_id"].astype(str).eq(source_experiment_id)
        & designs["design_id"].astype(str).eq(design_id)
        & designs["reduction_id"].astype(str).eq(reduction_id)
    ]
    if len(selected) != 1:
        raise ValueError(
            "four-state event-window diagnostic requires exactly one designs row for "
            f"source experiment {source_experiment_id!r}, design {design_id!r}, and reduction {reduction_id!r}; "
            f"found {len(selected)}"
        )
    row = selected.iloc[0]
    response_basis = _choice(row["response_basis"], field="response_basis", choices={"post_window", "post_minus_pre"})
    pre_window = _pre_window(
        response_basis=response_basis,
        pre_window_duration_h=pre_window_duration_h,
        event_time_uncertainty_h=float(row["event_time_uncertainty_h"]),
    )
    window = (float(row["window_start_event_h"]), float(row["window_end_event_h"]))
    if not np.isfinite(window).all() or window[1] <= window[0]:
        raise ValueError("four-state event-window diagnostic requires a finite window start before its end")

    values = tuple(float(row[column]) for column in COMPONENT_COLUMNS)
    interval_low = tuple(float(row[f"{column}_descriptive_interval_low"]) for column in COMPONENT_COLUMNS)
    interval_high = tuple(float(row[f"{column}_descriptive_interval_high"]) for column in COMPONENT_COLUMNS)
    event_half_range = tuple(float(row[f"{column}_event_half_range"]) for column in COMPONENT_COLUMNS)
    if not np.isfinite((*values, *interval_low, *interval_high, *event_half_range)).all():
        raise ValueError("four-state event-window diagnostic requires finite component dispersion values")
    if any(low > high for low, high in zip(interval_low, interval_high, strict=True)):
        raise ValueError(
            "four-state event-window diagnostic descriptive-interval lower bounds must not exceed upper bounds"
        )
    if any(value < 0.0 for value in event_half_range):
        raise ValueError("four-state event-window diagnostic event-sensitivity half-ranges must be non-negative")

    trace_set = _select_traces(
        traces,
        source_experiment_id=source_experiment_id,
        design_id=design_id,
        reference_design_id=_identifier(row["reference_design_id"], field="reference_design_id"),
    )
    return FourStateEventWindowDiagnostic(
        source_experiment_id=source_experiment_id,
        design_id=design_id,
        reference_design_id=_identifier(row["reference_design_id"], field="reference_design_id"),
        reduction_id=reduction_id,
        reduction_method=_identifier(row["reduction_method"], field="reduction_method"),
        response_basis=response_basis,
        observation_stat=_choice(row["observation_stat"], field="observation_stat", choices={"mean", "median"}),
        descriptive_resampling_draws=_positive_int(
            row["descriptive_resampling_draws"], field="descriptive_resampling_draws"
        ),
        descriptive_interval_mass=_descriptive_interval_mass(row["descriptive_interval_mass"]),
        event_id=_identifier(row["event_id"], field="event_id"),
        event_time_uncertainty_h=_nonnegative_float(row["event_time_uncertainty_h"], field="event_time_uncertainty_h"),
        window=window,
        pre_window=pre_window,
        component_values=values,
        component_descriptive_interval_low=interval_low,
        component_descriptive_interval_high=interval_high,
        component_event_half_range=event_half_range,
        component_bound_kinds=tuple(
            _choice(row[f"{column}_bound_kind"], field=f"{column}_bound_kind", choices=BOUND_KINDS)
            for column in COMPONENT_COLUMNS
        ),
        component_has_policy_clipping=_component_bools(row, suffix="has_policy_clipping"),
        component_has_instrument_overflow=_component_bools(row, suffix="has_instrument_overflow"),
        component_event_has_policy_clipping=_component_bools(row, suffix="event_sensitivity_has_policy_clipping"),
        component_event_has_instrument_overflow=_component_bools(
            row, suffix="event_sensitivity_has_instrument_overflow"
        ),
        traces=trace_set,
    )


def _design_columns() -> set[str]:
    component_columns = {
        f"{component}_{suffix}"
        for component in COMPONENT_COLUMNS
        for suffix in (
            "descriptive_interval_low",
            "descriptive_interval_high",
            "event_half_range",
            "bound_kind",
            "has_policy_clipping",
            "has_instrument_overflow",
            "event_sensitivity_has_policy_clipping",
            "event_sensitivity_has_instrument_overflow",
        )
    }
    return {
        "experiment_id",
        "design_id",
        "reference_design_id",
        "reduction_id",
        "reduction_method",
        "response_basis",
        "observation_stat",
        "descriptive_resampling_draws",
        "descriptive_interval_mass",
        "event_id",
        "event_time_uncertainty_h",
        "window_start_event_h",
        "window_end_event_h",
        *COMPONENT_COLUMNS,
        *component_columns,
    }


def _select_traces(
    traces: pd.DataFrame,
    *,
    source_experiment_id: str,
    design_id: str,
    reference_design_id: str,
) -> pd.DataFrame:
    _require_columns(
        traces,
        {
            "experiment_id",
            "design_id",
            "position",
            "state",
            "time_from_event_h",
            "value",
            "value_policy_clipped",
            "value_instrument_overflow",
            "value_bound_kind",
            "signal_kind",
        },
        record="traces",
    )
    source = traces.loc[traces["experiment_id"].astype(str).eq(source_experiment_id)]
    selected = source.loc[source["design_id"].astype(str).eq(design_id)]
    _validate_trace_support(selected, design_id=design_id, signal_kinds=("growth", "response", "magnitude"))
    frames = [selected]
    if reference_design_id != design_id:
        reference = source.loc[
            source["design_id"].astype(str).eq(reference_design_id) & source["signal_kind"].astype(str).eq("magnitude")
        ]
        _validate_trace_support(reference, design_id=reference_design_id, signal_kinds=("magnitude",))
        frames.append(reference)
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["signal_kind", "design_id", "state", "position", "time_from_event_h"], kind="stable"
    )
    numeric = result.loc[:, ["time_from_event_h", "value"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("four-state event-window diagnostic traces require finite event-relative times and values")
    log_rows = result["signal_kind"].astype(str).isin({"response", "magnitude"})
    if result.loc[log_rows, "value"].astype(float).le(0.0).any():
        raise ValueError("four-state event-window diagnostic response and magnitude traces must be positive")
    unknown_bounds = sorted(set(result["value_bound_kind"].astype(str)) - BOUND_KINDS)
    if unknown_bounds:
        raise ValueError(f"four-state event-window diagnostic traces contain unsupported bounds: {unknown_bounds}")
    return result.reset_index(drop=True)


def _pre_window(
    *,
    response_basis: str,
    pre_window_duration_h: float | None,
    event_time_uncertainty_h: float,
) -> tuple[float, float] | None:
    uncertainty = _nonnegative_float(event_time_uncertainty_h, field="event_time_uncertainty_h")
    if response_basis == "post_window":
        if pre_window_duration_h is not None:
            raise ValueError("post_window diagnostics must not declare pre_window_duration_h")
        return None
    duration = _nonnegative_float(pre_window_duration_h, field="pre_window_duration_h")
    if duration <= 0.0:
        raise ValueError("post_minus_pre diagnostics require positive pre_window_duration_h")
    end = -uncertainty
    return (end - duration, end)


def _component_bools(row: pd.Series, *, suffix: str) -> tuple[bool, ...]:
    values: list[bool] = []
    for component in COMPONENT_COLUMNS:
        value = row[f"{component}_{suffix}"]
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"four-state event-window diagnostic {component}_{suffix} must be boolean")
        values.append(bool(value))
    return tuple(values)


def _validate_trace_support(rows: pd.DataFrame, *, design_id: str, signal_kinds: tuple[str, ...]) -> None:
    for signal_kind in signal_kinds:
        states = set(rows.loc[rows["signal_kind"].astype(str).eq(signal_kind), "state"].astype(str))
        if states != set(STATE_ORDER):
            raise ValueError(
                f"four-state event-window diagnostic requires {signal_kind!r} traces for design {design_id!r} "
                f"in states {list(STATE_ORDER)}; found {sorted(states)}"
            )


def _require_columns(frame: pd.DataFrame, required: set[str], *, record: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"four-state event-window diagnostic {record} record is missing columns: {missing}")


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"four-state event-window diagnostic {field} must be a non-empty string")
    return value.strip()


def _choice(value: object, *, field: str, choices: set[str] | frozenset[str]) -> str:
    result = _identifier(value, field=field)
    if result not in choices:
        raise ValueError(f"four-state event-window diagnostic {field} must be one of {sorted(choices)}")
    return result


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"four-state event-window diagnostic {field} must be a positive integer")
    return int(value)


def _nonnegative_float(value: object, *, field: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"four-state event-window diagnostic {field} must be a non-negative finite number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"four-state event-window diagnostic {field} must be a non-negative finite number")
    return result


def _descriptive_interval_mass(value: object) -> float:
    result = _nonnegative_float(value, field="descriptive_interval_mass")
    if not 0.5 < result < 1.0:
        raise ValueError("four-state event-window diagnostic descriptive_interval_mass must be between 0.5 and 1.0")
    return result


__all__ = ["FourStateEventWindowDiagnostic", "prepare_four_state_event_window_diagnostic"]
