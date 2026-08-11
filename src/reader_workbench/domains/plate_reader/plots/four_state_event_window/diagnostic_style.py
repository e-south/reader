"""Shared visual vocabulary for four-state event-window diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

STATE_COLORS = {
    "00": "#596675",
    "10": "#4F8C83",
    "01": "#607CC8",
    "11": "#C15D6E",
}
STATE_MARKERS = {"00": "o", "10": "s", "01": "^", "11": "D"}
BOUND_MARKERS = {"exact": "o", "lower": ">", "upper": "<", "indeterminate": "X"}

_DEFAULT_AXIS_LABELS = {
    "growth": "Signal",
    "response": "log$_2$(response signal)",
    "magnitude": "log$_2$(magnitude signal)",
}


def validated_axis_labels(axis_labels: Mapping[str, str] | None) -> dict[str, str]:
    """Validate the three compiler-owned signal labels."""

    labels = dict(_DEFAULT_AXIS_LABELS if axis_labels is None else axis_labels)
    if set(labels) != set(_DEFAULT_AXIS_LABELS):
        raise ValueError("four-state event-window diagnostic axis labels must define growth, response, and magnitude")
    if any(not isinstance(value, str) or not value.strip() for value in labels.values()):
        raise ValueError("four-state event-window diagnostic axis labels must be non-empty strings")
    return {key: value.strip() for key, value in labels.items()}


def validated_state_labels(state_labels: Mapping[str, str] | None) -> dict[str, str]:
    """Validate one display label for every canonical state."""

    labels = {state: state for state in STATE_MARKERS} if state_labels is None else dict(state_labels)
    if set(labels) != set(STATE_MARKERS):
        raise ValueError("four-state event-window diagnostic state labels must define 00, 10, 01, and 11")
    if any(not isinstance(value, str) or not value.strip() for value in labels.values()):
        raise ValueError("four-state event-window diagnostic state labels must be non-empty strings")
    return {key: value.strip() for key, value in labels.items()}


__all__ = [
    "BOUND_MARKERS",
    "STATE_COLORS",
    "STATE_MARKERS",
    "validated_axis_labels",
    "validated_state_labels",
]
