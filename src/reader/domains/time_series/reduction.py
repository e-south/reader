"""Pure temporal trace reduction with explicit selection and support policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .contracts import EndpointSelection, IntervalSelection, TemporalReductionSpec

ValueBoundKind = Literal["exact", "lower", "upper", "indeterminate"]
_EXACT_TIME_ATOL_H = 1.0e-9


@dataclass(frozen=True)
class TemporalReductionResult:
    value: float
    observed_point_count: int
    evaluation_point_count: int
    max_interior_gap_h: float
    policy_clipped_point_count: int
    instrument_overflow_point_count: int
    bound_kind: ValueBoundKind


def reduce_temporal_trace(
    times: np.ndarray,
    values: np.ndarray,
    *,
    spec: TemporalReductionSpec,
    trace_id: str,
    origin_h: float | None = None,
    policy_clipped: np.ndarray | None = None,
    instrument_overflow: np.ndarray | None = None,
    bound_kinds: np.ndarray | None = None,
) -> TemporalReductionResult:
    """Reduce one trace without extrapolation or implicit time semantics."""

    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
        raise ValueError(f"{trace_id} must contain aligned one-dimensional time and value arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{trace_id} contains non-finite time or signal values")

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    if len(np.unique(x)) != len(x):
        raise ValueError(f"{trace_id} contains duplicate time coordinates")
    policy, overflow, bounds = _value_provenance(
        size=len(x),
        order=order,
        policy_clipped=policy_clipped,
        instrument_overflow=instrument_overflow,
        bound_kinds=bound_kinds,
        trace_id=trace_id,
    )
    origin = _resolved_origin(spec=spec, origin_h=origin_h)
    if isinstance(spec.selection, EndpointSelection):
        return _reduce_endpoint(
            x,
            y,
            policy=policy,
            overflow=overflow,
            bounds=bounds,
            spec=spec,
            endpoint_h=origin + spec.selection.time_h,
            trace_id=trace_id,
        )
    return _reduce_interval(
        x,
        y,
        policy=policy,
        overflow=overflow,
        bounds=bounds,
        spec=spec,
        window_start_h=origin + spec.selection.start_h,
        window_end_h=origin + spec.selection.end_h,
        trace_id=trace_id,
    )


def _resolved_origin(*, spec: TemporalReductionSpec, origin_h: float | None) -> float:
    selection = spec.selection
    if selection.time_basis == "absolute":
        if origin_h is not None and not math.isclose(float(origin_h), 0.0, abs_tol=0.0):
            raise ValueError("absolute temporal reductions cannot declare a nonzero origin_h")
        return 0.0
    if origin_h is None or not math.isfinite(float(origin_h)):
        raise ValueError("event-relative temporal reductions require a finite origin_h")
    return float(origin_h)


def _reduce_endpoint(
    x: np.ndarray,
    y: np.ndarray,
    *,
    policy: np.ndarray,
    overflow: np.ndarray,
    bounds: np.ndarray,
    spec: TemporalReductionSpec,
    endpoint_h: float,
    trace_id: str,
) -> TemporalReductionResult:
    selection = spec.selection
    assert isinstance(selection, EndpointSelection)
    delta = np.abs(x - endpoint_h)
    if selection.mode == "exact":
        candidates = np.flatnonzero(delta <= _EXACT_TIME_ATOL_H)
        missing_message = f"{trace_id} has no exact observation at {endpoint_h:g} h"
    else:
        minimum = float(delta.min())
        candidates = np.flatnonzero(np.isclose(delta, minimum, rtol=0.0, atol=1.0e-12))
        missing_message = (
            f"{trace_id} has no endpoint observation within {selection.tolerance_h:g} h of {endpoint_h:g} h"
        )
        if minimum > selection.tolerance_h:
            candidates = np.asarray([], dtype=int)
    if len(candidates) == 0:
        raise ValueError(missing_message)
    if len(candidates) != 1:
        raise ValueError(f"{trace_id} endpoint selection is ambiguous at {endpoint_h:g} h")
    index = int(candidates[0])
    positive_values = y if spec.support.positive_value_scope == "entire_trace" else y[index : index + 1]
    _require_positive(positive_values, spec=spec, trace_id=trace_id)
    _reject_censored(
        policy=policy[index : index + 1],
        overflow=overflow[index : index + 1],
        spec=spec,
        trace_id=trace_id,
    )
    value = _project_output(float(y[index]), spec=spec, trace_id=trace_id)
    return TemporalReductionResult(
        value=value,
        observed_point_count=1,
        evaluation_point_count=1,
        max_interior_gap_h=0.0,
        policy_clipped_point_count=int(policy[index]),
        instrument_overflow_point_count=int(overflow[index]),
        bound_kind=combine_bound_kinds(bounds[index]),
    )


def _reduce_interval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    policy: np.ndarray,
    overflow: np.ndarray,
    bounds: np.ndarray,
    spec: TemporalReductionSpec,
    window_start_h: float,
    window_end_h: float,
    trace_id: str,
) -> TemporalReductionResult:
    selection = spec.selection
    assert isinstance(selection, IntervalSelection)
    if window_end_h <= window_start_h:
        raise ValueError(f"{trace_id} interval resolves to a non-increasing absolute window")

    support = spec.support
    reduction_x = _normalized_interval_times(
        x,
        window_start_h=window_start_h,
        window_end_h=window_end_h,
        boundary_support=support.boundary_support,
        trace_id=trace_id,
    )
    selected = (reduction_x >= window_start_h) & (reduction_x <= window_end_h)
    selected_indexes = np.flatnonzero(selected)
    if len(selected_indexes) < support.minimum_observations:
        raise ValueError(
            f"{trace_id} has {len(selected_indexes)} observations in the selected interval; "
            f"at least {support.minimum_observations} are required"
        )
    left_index, right_index = _support_bounds(
        reduction_x,
        selected_indexes=selected_indexes,
        window_start_h=window_start_h,
        window_end_h=window_end_h,
        boundary_support=support.boundary_support,
        trace_id=trace_id,
    )
    support_slice = slice(left_index, right_index + 1)
    support_x = reduction_x[support_slice]
    observed_max_gap = float(np.max(np.diff(support_x))) if len(support_x) > 1 else 0.0
    if support.maximum_interior_gap_h is not None and observed_max_gap > support.maximum_interior_gap_h:
        raise ValueError(f"{trace_id} interior gap {observed_max_gap:g} h exceeds {support.maximum_interior_gap_h:g} h")
    _reject_censored(
        policy=policy[support_slice],
        overflow=overflow[support_slice],
        spec=spec,
        trace_id=trace_id,
    )
    positive_values = y if support.positive_value_scope == "entire_trace" else y[support_slice]
    _require_positive(positive_values, spec=spec, trace_id=trace_id)

    method = spec.method
    evaluation_count: int
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if method == "observed_mean":
            reduced = float(np.mean(y[selected]))
            evaluation_count = len(selected_indexes)
        elif method == "observed_median":
            reduced = float(np.median(y[selected]))
            evaluation_count = len(selected_indexes)
        else:
            inside = (reduction_x > window_start_h) & (reduction_x < window_end_h)
            window_x = np.concatenate(([window_start_h], reduction_x[inside], [window_end_h]))
            duration = float(window_end_h - window_start_h)
            if method == "geometric_time_mean":
                log_values = np.log2(y)
                window_values = np.concatenate(
                    (
                        [np.interp(window_start_h, reduction_x, log_values)],
                        log_values[inside],
                        [np.interp(window_end_h, reduction_x, log_values)],
                    )
                )
                reduced = float(np.trapezoid(window_values, window_x) / duration)
            elif method == "integrated_linear_mean":
                window_values = np.concatenate(
                    (
                        [np.interp(window_start_h, reduction_x, y)],
                        y[inside],
                        [np.interp(window_end_h, reduction_x, y)],
                    )
                )
                reduced = float(np.log2(float(np.trapezoid(window_values, window_x) / duration)))
            else:  # pragma: no cover - TemporalReductionSpec validates this pair.
                raise ValueError(f"unsupported interval temporal reduction method: {method!r}")
            evaluation_count = len(window_x)
    if method in {"observed_mean", "observed_median"}:
        reduced = _project_output(reduced, spec=spec, trace_id=trace_id)
    if not math.isfinite(reduced):
        raise ValueError(f"{trace_id} temporal reduction produced a non-finite value")

    return TemporalReductionResult(
        value=reduced,
        observed_point_count=len(selected_indexes),
        evaluation_point_count=evaluation_count,
        max_interior_gap_h=observed_max_gap,
        policy_clipped_point_count=int(policy[support_slice].sum()),
        instrument_overflow_point_count=int(overflow[support_slice].sum()),
        bound_kind=combine_bound_kinds(*bounds[support_slice]),
    )


def _normalized_interval_times(
    x: np.ndarray,
    *,
    window_start_h: float,
    window_end_h: float,
    boundary_support: str,
    trace_id: str,
) -> np.ndarray:
    if boundary_support != "observed":
        return x
    normalized = x.copy()
    start_indexes = np.flatnonzero(np.isclose(x, window_start_h, rtol=0.0, atol=_EXACT_TIME_ATOL_H))
    end_indexes = np.flatnonzero(np.isclose(x, window_end_h, rtol=0.0, atol=_EXACT_TIME_ATOL_H))
    if len(start_indexes) > 1:
        raise ValueError(f"{trace_id} has ambiguous observations at the interval start {window_start_h:g} h")
    if len(end_indexes) > 1:
        raise ValueError(f"{trace_id} has ambiguous observations at the interval end {window_end_h:g} h")
    if len(start_indexes) == 1 and len(end_indexes) == 1 and start_indexes[0] == end_indexes[0]:
        raise ValueError(f"{trace_id} has one observation matching both interval boundaries")
    if len(start_indexes) == 1:
        normalized[start_indexes[0]] = window_start_h
    if len(end_indexes) == 1:
        normalized[end_indexes[0]] = window_end_h
    return normalized


def _support_bounds(
    x: np.ndarray,
    *,
    selected_indexes: np.ndarray,
    window_start_h: float,
    window_end_h: float,
    boundary_support: str,
    trace_id: str,
) -> tuple[int, int]:
    if boundary_support == "covered":
        if x[0] > window_start_h or x[-1] < window_end_h:
            raise ValueError(f"{trace_id} does not cover [{window_start_h:g}, {window_end_h:g}] without extrapolation")
        left_index = max(int(np.searchsorted(x, window_start_h, side="right")) - 1, 0)
        right_index = min(int(np.searchsorted(x, window_end_h, side="left")), len(x) - 1)
        return left_index, right_index
    if boundary_support == "observed" and (
        not np.any(np.isclose(x, window_start_h, rtol=0.0, atol=_EXACT_TIME_ATOL_H))
        or not np.any(np.isclose(x, window_end_h, rtol=0.0, atol=_EXACT_TIME_ATOL_H))
    ):
        raise ValueError(
            f"{trace_id} requires observed interval boundaries at {window_start_h:g} and {window_end_h:g} h"
        )
    if len(selected_indexes) == 0:
        raise ValueError(f"{trace_id} selected interval contains no observations")
    return int(selected_indexes[0]), int(selected_indexes[-1])


def _project_output(value: float, *, spec: TemporalReductionSpec, trace_id: str) -> float:
    if spec.output_space == "linear":
        return value
    floor = spec.support.positive_floor
    assert floor is not None
    if value <= floor:
        raise ValueError(f"{trace_id} reduced value is at or below the positive floor {floor:g}")
    return float(np.log2(value))


def _require_positive(values: np.ndarray, *, spec: TemporalReductionSpec, trace_id: str) -> None:
    floor = spec.support.positive_floor
    if floor is not None and np.any(values <= floor):
        raise ValueError(
            f"{trace_id} contains values at or below the positive floor {floor:g}; "
            "log-space reduction requires strictly positive values"
        )


def _reject_censored(
    *,
    policy: np.ndarray,
    overflow: np.ndarray,
    spec: TemporalReductionSpec,
    trace_id: str,
) -> None:
    if spec.support.censored_values == "reject" and np.any(policy | overflow):
        raise ValueError(f"{trace_id} selected support contains clipped or overflowed observations")


def _value_provenance(
    *,
    size: int,
    order: np.ndarray,
    policy_clipped: np.ndarray | None,
    instrument_overflow: np.ndarray | None,
    bound_kinds: np.ndarray | None,
    trace_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy = np.zeros(size, dtype=bool) if policy_clipped is None else np.asarray(policy_clipped)
    overflow = np.zeros(size, dtype=bool) if instrument_overflow is None else np.asarray(instrument_overflow)
    bounds = np.full(size, "exact", dtype=object) if bound_kinds is None else np.asarray(bound_kinds, dtype=object)
    for name, values in (("policy clipping", policy), ("instrument overflow", overflow), ("bounds", bounds)):
        if values.ndim != 1 or len(values) != size:
            raise ValueError(f"{trace_id} {name} provenance must align with time and value arrays")
    if policy.dtype.kind != "b" or overflow.dtype.kind != "b":
        raise ValueError(f"{trace_id} clipping and overflow provenance must be boolean")
    allowed = {"exact", "lower", "upper", "indeterminate"}
    unknown = sorted(set(map(str, bounds)) - allowed)
    if unknown:
        raise ValueError(f"{trace_id} contains unsupported value bounds: {unknown}")
    if np.any((policy | overflow) & (bounds.astype(str) == "exact")):
        raise ValueError(f"{trace_id} marks clipped or overflowed observations as exact")
    return policy[order], overflow[order], bounds[order]


def combine_bound_kinds(*kinds: object) -> ValueBoundKind:
    active = {str(kind) for kind in kinds} - {"exact"}
    if not active:
        return "exact"
    if "indeterminate" in active or len(active) > 1:
        return "indeterminate"
    result = active.pop()
    if result not in {"lower", "upper"}:  # pragma: no cover - source validation owns this boundary.
        raise ValueError(f"unsupported value bound kind: {result!r}")
    return result  # type: ignore[return-value]


def invert_bound_kind(kind: object) -> ValueBoundKind:
    inverse: dict[str, ValueBoundKind] = {
        "exact": "exact",
        "lower": "upper",
        "upper": "lower",
        "indeterminate": "indeterminate",
    }
    try:
        return inverse[str(kind)]
    except KeyError as exc:  # pragma: no cover - source validation owns this boundary.
        raise ValueError(f"unsupported value bound kind: {kind!r}") from exc


__all__ = [
    "TemporalReductionResult",
    "ValueBoundKind",
    "combine_bound_kinds",
    "invert_bound_kind",
    "reduce_temporal_trace",
]
