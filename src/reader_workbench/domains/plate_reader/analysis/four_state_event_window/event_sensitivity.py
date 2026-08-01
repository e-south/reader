"""Event-time sensitivity primitives for four-state event-window reductions."""

from __future__ import annotations

import numpy as np


def symmetric_event_sensitivity_half_width(
    midpoint: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """Return a midpoint-centered envelope covering both event-bound reductions.

    Four-state event-window reduction is not necessarily linear in event time. The
    midpoint estimate therefore need not bisect the values obtained at the two
    event bounds. A symmetric display interval uses the larger absolute
    deviation from the midpoint.
    """

    values = np.asarray([midpoint, lower_bound, upper_bound], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("event-time sensitivity values must be finite.")
    return float(max(abs(values[0] - values[1]), abs(values[2] - values[0])))


__all__ = ["symmetric_event_sensitivity_half_width"]
