"""Numerical reduction primitives for event-relative signal trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .contracts import ReductionMethod

ValueBoundKind = Literal["exact", "lower", "upper", "indeterminate"]


@dataclass(frozen=True)
class TraceSummary:
    value: float
    observed_point_count: int
    integration_point_count: int
    max_interior_gap_h: float
    policy_clipped_point_count: int
    instrument_overflow_point_count: int
    bound_kind: ValueBoundKind


@dataclass(frozen=True)
class ValueProvenanceSummary:
    policy_clipped_point_count: int
    instrument_overflow_point_count: int
    bound_kind: ValueBoundKind


def summarize_trace(
    times: np.ndarray,
    values: np.ndarray,
    *,
    window_start_h: float,
    window_end_h: float,
    method: ReductionMethod,
    positive_floor: float,
    max_interior_gap_h: float,
    trace_id: str,
    policy_clipped: np.ndarray | None = None,
    instrument_overflow: np.ndarray | None = None,
    bound_kinds: np.ndarray | None = None,
) -> TraceSummary:
    """Reduce one signal trace without extrapolation or value clipping."""

    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError(f"{trace_id} must contain aligned one-dimensional time and value arrays.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{trace_id} contains non-finite time or signal values.")
    if not np.isfinite(window_start_h) or not np.isfinite(window_end_h) or window_end_h <= window_start_h:
        raise ValueError("trace reduction requires finite window_start_h < window_end_h.")
    if positive_floor <= 0.0 or max_interior_gap_h <= 0.0:
        raise ValueError("trace reduction floors and gap limits must be positive.")

    order, x, policy, overflow, bounds, left_index, right_index = _window_value_provenance(
        times=x,
        window_start_h=window_start_h,
        window_end_h=window_end_h,
        policy_clipped=policy_clipped,
        instrument_overflow=instrument_overflow,
        bound_kinds=bound_kinds,
        trace_id=trace_id,
    )
    y = y[order]
    if np.any(y <= positive_floor):
        raise ValueError(
            f"{trace_id} contains values at or below the positive floor {positive_floor:g}; "
            "log-space reduction requires strictly positive values."
        )

    support = (x >= window_start_h) & (x <= window_end_h)
    support_indexes = np.flatnonzero(support)
    gap_slice = x[left_index : right_index + 1]
    observed_max_gap = float(np.max(np.diff(gap_slice))) if len(gap_slice) > 1 else float("inf")
    if observed_max_gap > max_interior_gap_h:
        raise ValueError(f"{trace_id} interior gap {observed_max_gap:g} h exceeds {max_interior_gap_h:g} h.")

    inside = (x > window_start_h) & (x < window_end_h)
    window_x = np.concatenate(([window_start_h], x[inside], [window_end_h]))
    duration = float(window_end_h - window_start_h)
    if method == "geometric_time_mean":
        log_values = np.log2(y)
        window_values = np.concatenate(
            (
                [np.interp(window_start_h, x, log_values)],
                log_values[inside],
                [np.interp(window_end_h, x, log_values)],
            )
        )
        reduced = float(np.trapezoid(window_values, window_x) / duration)
    elif method == "integrated_linear_mean":
        window_values = np.concatenate(
            (
                [np.interp(window_start_h, x, y)],
                y[inside],
                [np.interp(window_end_h, x, y)],
            )
        )
        reduced = float(np.log2(float(np.trapezoid(window_values, window_x) / duration)))
    else:  # pragma: no cover - validated by ReductionSpec.
        raise ValueError(f"unsupported reduction method: {method!r}.")

    return TraceSummary(
        value=reduced,
        observed_point_count=int(len(support_indexes)),
        integration_point_count=int(len(window_x)),
        max_interior_gap_h=observed_max_gap,
        policy_clipped_point_count=int(policy[left_index : right_index + 1].sum()),
        instrument_overflow_point_count=int(overflow[left_index : right_index + 1].sum()),
        bound_kind=combine_bound_kinds(*bounds[left_index : right_index + 1]),
    )


def summarize_value_provenance(
    times: np.ndarray,
    *,
    window_start_h: float,
    window_end_h: float,
    policy_clipped: np.ndarray,
    instrument_overflow: np.ndarray,
    bound_kinds: np.ndarray,
    trace_id: str,
) -> ValueProvenanceSummary:
    """Summarize the observations supporting one interpolated window."""

    _, _, policy, overflow, bounds, left_index, right_index = _window_value_provenance(
        times=times,
        window_start_h=window_start_h,
        window_end_h=window_end_h,
        policy_clipped=policy_clipped,
        instrument_overflow=instrument_overflow,
        bound_kinds=bound_kinds,
        trace_id=trace_id,
    )
    return ValueProvenanceSummary(
        policy_clipped_point_count=int(policy[left_index : right_index + 1].sum()),
        instrument_overflow_point_count=int(overflow[left_index : right_index + 1].sum()),
        bound_kind=combine_bound_kinds(*bounds[left_index : right_index + 1]),
    )


def combine_bound_kinds(*kinds: object) -> ValueBoundKind:
    active = {str(kind) for kind in kinds} - {"exact"}
    if not active:
        return "exact"
    if "indeterminate" in active or len(active) > 1:
        return "indeterminate"
    result = active.pop()
    if result not in {"lower", "upper"}:  # pragma: no cover - callers validate before combining.
        raise ValueError(f"unsupported value bound kind: {result!r}.")
    return result  # type: ignore[return-value]


def invert_bound_kind(kind: object) -> ValueBoundKind:
    value = str(kind)
    inverse: dict[str, ValueBoundKind] = {
        "exact": "exact",
        "lower": "upper",
        "upper": "lower",
        "indeterminate": "indeterminate",
    }
    try:
        return inverse[value]
    except KeyError as exc:  # pragma: no cover - source loading validates this contract.
        raise ValueError(f"unsupported value bound kind: {value!r}.") from exc


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
            raise ValueError(f"{trace_id} {name} provenance must align with time and value arrays.")
    if policy.dtype.kind != "b" or overflow.dtype.kind != "b":
        raise ValueError(f"{trace_id} clipping and overflow provenance must be boolean.")
    allowed = {"exact", "lower", "upper", "indeterminate"}
    unknown = sorted(set(map(str, bounds)) - allowed)
    if unknown:
        raise ValueError(f"{trace_id} contains unsupported value bounds: {unknown}.")
    if np.any((policy | overflow) & (bounds.astype(str) == "exact")):
        raise ValueError(f"{trace_id} marks clipped or overflowed observations as exact.")
    return policy[order], overflow[order], bounds[order]


def _window_value_provenance(
    *,
    times: np.ndarray,
    window_start_h: float,
    window_end_h: float,
    policy_clipped: np.ndarray | None,
    instrument_overflow: np.ndarray | None,
    bound_kinds: np.ndarray | None,
    trace_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    x = np.asarray(times, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError(f"{trace_id} must contain at least two finite time values.")
    if not np.isfinite(window_start_h) or not np.isfinite(window_end_h) or window_end_h <= window_start_h:
        raise ValueError("trace provenance requires finite window_start_h < window_end_h.")
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    policy, overflow, bounds = _value_provenance(
        size=len(x),
        order=order,
        policy_clipped=policy_clipped,
        instrument_overflow=instrument_overflow,
        bound_kinds=bound_kinds,
        trace_id=trace_id,
    )
    if np.any(np.diff(x) == 0.0):
        raise ValueError(f"{trace_id} contains duplicate time values.")
    if x[0] > window_start_h or x[-1] < window_end_h:
        raise ValueError(
            f"{trace_id} does not cover [{window_start_h:g}, {window_end_h:g}] h; observed [{x[0]:g}, {x[-1]:g}] h."
        )
    left_index = max(0, int(np.searchsorted(x, window_start_h, side="right") - 1))
    right_index = min(len(x) - 1, int(np.searchsorted(x, window_end_h, side="left")))
    return order, x, policy, overflow, bounds, left_index, right_index


__all__ = [
    "TraceSummary",
    "ValueProvenanceSummary",
    "ValueBoundKind",
    "combine_bound_kinds",
    "invert_bound_kind",
    "summarize_trace",
    "summarize_value_provenance",
]
