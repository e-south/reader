"""Numerical reduction primitives for event-relative ratio trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ReductionMethod


@dataclass(frozen=True)
class TraceSummary:
    value: float
    observed_point_count: int
    integration_point_count: int
    max_interior_gap_h: float


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
) -> TraceSummary:
    """Reduce one ratio trace without extrapolation or value clipping."""

    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError(f"{trace_id} must contain aligned one-dimensional time and value arrays.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{trace_id} contains non-finite time or ratio values.")
    if not np.isfinite(window_start_h) or not np.isfinite(window_end_h) or window_end_h <= window_start_h:
        raise ValueError("trace reduction requires finite window_start_h < window_end_h.")
    if positive_floor <= 0.0 or max_interior_gap_h <= 0.0:
        raise ValueError("trace reduction floors and gap limits must be positive.")

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    if np.any(np.diff(x) == 0.0):
        raise ValueError(f"{trace_id} contains duplicate time values.")
    if x[0] > window_start_h or x[-1] < window_end_h:
        raise ValueError(
            f"{trace_id} does not cover [{window_start_h:g}, {window_end_h:g}] h; observed [{x[0]:g}, {x[-1]:g}] h."
        )
    if np.any(y <= positive_floor):
        raise ValueError(f"{trace_id} contains ratio values at or below the positive floor {positive_floor:g}.")

    support = (x >= window_start_h) & (x <= window_end_h)
    support_indexes = np.flatnonzero(support)
    left_index = max(0, int(np.searchsorted(x, window_start_h, side="right") - 1))
    right_index = min(len(x) - 1, int(np.searchsorted(x, window_end_h, side="left")))
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
    )


__all__ = ["TraceSummary", "summarize_trace"]
