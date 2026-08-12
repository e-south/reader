"""Render the reduced components of a four-state event-window diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .diagnostic import FourStateEventWindowDiagnostic
from .diagnostic_style import BOUND_MARKERS, STATE_COLORS
from .schema import COMPONENT_COLUMNS


def draw_component_panel(
    axis: Any,
    *,
    diagnostic: FourStateEventWindowDiagnostic,
    axis_labels: Mapping[str, str],
    reference_label: str,
) -> None:
    """Draw reduced components and their two semantic groups."""

    y = np.arange(len(COMPONENT_COLUMNS))
    values = np.asarray(diagnostic.component_values)
    for index, (component, value) in enumerate(zip(COMPONENT_COLUMNS, values, strict=True)):
        state = component[1:]
        color = STATE_COLORS[state]
        axis.scatter(
            value,
            y[index],
            color=color,
            marker=BOUND_MARKERS[diagnostic.component_bound_kinds[index]],
            s=30.0,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    axis.axvline(0.0, color="#64748b", linewidth=0.9)
    axis.axhline(3.5, color="#cbd5e1", linewidth=0.8)
    axis.set_yticks(y, labels=COMPONENT_COLUMNS)
    axis.invert_yaxis()
    axis.set_xlabel("Reduced value (log$_2$ units)", labelpad=8)
    _draw_group_label(axis, start=-0.35, end=3.35, label=f"Response $r_i$\n{axis_labels['response']}")
    _draw_group_label(
        axis,
        start=3.65,
        end=7.35,
        label=f"Signal $b_i$\n{axis_labels['magnitude']}\nrelative to {reference_label}",
    )
    notes = quality_notes(diagnostic)
    if notes:
        axis.text(
            0.0,
            -0.16,
            "Quality flags — " + "; ".join(notes),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color="#7c2d12",
            wrap=True,
        )


def _draw_group_label(axis: Any, *, start: float, end: float, label: str) -> None:
    transform = axis.get_yaxis_transform()
    (bracket,) = axis.plot(
        [1.025, 1.055, 1.055, 1.025],
        [start, start, end, end],
        color="#8793A1",
        linewidth=0.9,
        transform=transform,
        clip_on=False,
    )
    bracket.set_gid("four-state-event-window-component-group")
    axis.text(
        1.075,
        (start + end) / 2.0,
        label,
        transform=transform,
        ha="left",
        va="center",
        fontsize=11.5,
        color="#536273",
        linespacing=1.15,
        clip_on=False,
    )


def quality_notes(diagnostic: FourStateEventWindowDiagnostic) -> list[str]:
    notes: list[str] = []
    for index, component in enumerate(COMPONENT_COLUMNS):
        flags: list[str] = []
        bound = diagnostic.component_bound_kinds[index]
        if bound != "exact":
            flags.append(f"{bound} bound")
        if diagnostic.component_has_policy_clipping[index]:
            flags.append("policy clipping")
        if diagnostic.component_has_instrument_overflow[index]:
            flags.append("instrument overflow")
        if flags:
            notes.append(f"{component}: {', '.join(flags)}")
    return notes


def has_quality_flags(diagnostic: FourStateEventWindowDiagnostic) -> bool:
    trace_flags = (
        diagnostic.traces["value_policy_clipped"].astype(bool)
        | diagnostic.traces["value_instrument_overflow"].astype(bool)
        | diagnostic.traces["value_bound_kind"].astype(str).ne("exact")
    ).any()
    return bool(trace_flags or quality_notes(diagnostic))


__all__ = ["draw_component_panel", "has_quality_flags", "quality_notes"]
