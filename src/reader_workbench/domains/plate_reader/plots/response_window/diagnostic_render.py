"""Render a provenance-aware response-window diagnostic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from reader_workbench.plotting.style import use_style

from .diagnostic import ResponseWindowDiagnostic, prepare_response_window_diagnostic
from .schema import COMPONENT_COLUMNS, STATE_ORDER

STATE_COLORS = {
    "00": "#334155",
    "10": "#0f766e",
    "01": "#2563eb",
    "11": "#be123c",
}
BOUND_MARKERS = {"exact": "o", "lower": ">", "upper": "<", "indeterminate": "X"}


def render_response_window_diagnostic(
    traces: pd.DataFrame,
    designs: pd.DataFrame,
    *,
    source_experiment_id: str,
    design_id: str,
    reduction_id: str,
    pre_window_duration_h: float | None,
    title: str | None = None,
) -> Any:
    """Render observed traces, descriptive dispersion, and event sensitivity in one row."""

    diagnostic = prepare_response_window_diagnostic(
        traces,
        designs,
        source_experiment_id=source_experiment_id,
        design_id=design_id,
        reduction_id=reduction_id,
        pre_window_duration_h=pre_window_duration_h,
    )
    with use_style(
        {
            "axes_titleweight": "regular",
            "figure_figsize": (16.0, 5.2),
            "font_scale": 0.76,
        }
    ):
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.lines import Line2D  # noqa: PLC0415

        figure, axes = plt.subplots(
            1,
            4,
            figsize=(16.0, 5.2),
            constrained_layout=True,
            gridspec_kw={"width_ratios": (1.0, 1.0, 1.0, 0.95)},
        )
        figure.set_gid("response-window-diagnostic")
        for axis, signal_kind, panel_title, ylabel in (
            (axes[0], "growth", "Growth traces", "signal value"),
            (axes[1], "response", "Response traces", "log2 signal value"),
            (axes[2], "magnitude", "Magnitude traces + reference", "log2 signal value"),
        ):
            _draw_trace_panel(axis, diagnostic=diagnostic, signal_kind=signal_kind)
            axis.set_title(panel_title)
            axis.set_xlabel("Time from event estimate (h)")
            axis.set_ylabel(ylabel)
        _draw_component_panel(axes[3], diagnostic=diagnostic)
        axes[3].set_title("Reduced components")

        interval_mass_percent = diagnostic.descriptive_interval_mass * 100.0
        metadata = (
            f"{diagnostic.observation_stat} center across within-experiment observations when grids align · "
            f"{diagnostic.reduction_method} / {diagnostic.response_basis} · "
            f"{interval_mass_percent:g}% descriptive resampling interval "
            f"({diagnostic.descriptive_resampling_draws} draws) · "
            f"event estimate uncertainty ±{diagnostic.event_time_uncertainty_h:g} h"
        )
        figure.suptitle(f"{title or f'{diagnostic.source_experiment_id} :: {diagnostic.design_id}'}\n{metadata}")

        legend = [
            Line2D([0], [0], color=STATE_COLORS[state], marker="o", linewidth=1.8, label=state) for state in STATE_ORDER
        ]
        if diagnostic.reference_design_id != diagnostic.design_id:
            legend.append(Line2D([0], [0], color="#64748b", linestyle="--", linewidth=1.4, label="reference"))
        legend.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="#64748b",
                    linewidth=5.0,
                    alpha=0.25,
                    label="event-time sensitivity",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#64748b",
                    linewidth=1.5,
                    label=f"{interval_mass_percent:g}% descriptive resampling interval",
                ),
            ]
        )
        if _has_quality_flags(diagnostic):
            legend.append(Line2D([0], [0], color="#7c2d12", marker="x", linestyle="none", label="quality/bound flag"))
        figure.legend(handles=legend, loc="outside lower center", ncol=min(len(legend), 8), frameon=False)
    return figure


def _draw_trace_panel(axis: Any, *, diagnostic: ResponseWindowDiagnostic, signal_kind: str) -> None:
    rows = diagnostic.traces.loc[diagnostic.traces["signal_kind"].astype(str).eq(signal_kind)].copy()
    rows["plot_value"] = rows["value"].astype(float)
    if signal_kind in {"response", "magnitude"}:
        rows["plot_value"] = np.log2(rows["plot_value"])
    _draw_time_context(axis, diagnostic=diagnostic, signal_kind=signal_kind)
    for state in STATE_ORDER:
        selected = rows.loc[
            rows["design_id"].astype(str).eq(diagnostic.design_id) & rows["state"].astype(str).eq(state)
        ]
        _draw_observed_traces(
            axis,
            selected,
            color=STATE_COLORS[state],
            linestyle="-",
            observation_stat=diagnostic.observation_stat,
            gid="response-window-trace",
        )
        if signal_kind == "magnitude" and diagnostic.reference_design_id != diagnostic.design_id:
            reference = rows.loc[
                rows["design_id"].astype(str).eq(diagnostic.reference_design_id) & rows["state"].astype(str).eq(state)
            ]
            _draw_observed_traces(
                axis,
                reference,
                color=STATE_COLORS[state],
                linestyle="--",
                observation_stat=diagnostic.observation_stat,
                gid="response-window-reference-trace",
            )


def _draw_time_context(axis: Any, *, diagnostic: ResponseWindowDiagnostic, signal_kind: str) -> None:
    uncertainty = diagnostic.event_time_uncertainty_h
    if uncertainty > 0.0:
        interval = axis.axvspan(-uncertainty, uncertainty, color="#64748b", alpha=0.10, linewidth=0.0)
        interval.set_gid("response-window-event-interval")
    event = axis.axvline(0.0, color="#64748b", linewidth=0.9)
    event.set_gid("response-window-event-estimate")
    post = axis.axvspan(*diagnostic.window, color="#f0c36e", alpha=0.18, linewidth=0.0)
    post.set_gid("response-window-reduction-window")
    if signal_kind == "response" and diagnostic.pre_window is not None:
        pre = axis.axvspan(*diagnostic.pre_window, color="#94a3b8", alpha=0.14, linewidth=0.0)
        pre.set_gid("response-window-pre-window")


def _draw_observed_traces(
    axis: Any,
    rows: pd.DataFrame,
    *,
    color: str,
    linestyle: str,
    observation_stat: str,
    gid: str,
) -> None:
    traces: list[pd.DataFrame] = []
    for _, trace in rows.groupby("position", sort=True):
        trace = trace.sort_values("time_from_event_h", kind="stable")
        traces.append(trace)
        axis.plot(
            trace["time_from_event_h"],
            trace["plot_value"],
            color=color,
            linestyle=linestyle,
            linewidth=0.8,
            alpha=0.22,
        )
        flagged = trace.loc[
            trace["value_policy_clipped"].astype(bool)
            | trace["value_instrument_overflow"].astype(bool)
            | trace["value_bound_kind"].astype(str).ne("exact")
        ]
        if not flagged.empty:
            axis.scatter(
                flagged["time_from_event_h"],
                flagged["plot_value"],
                marker="x",
                s=18.0,
                color="#7c2d12",
                zorder=4,
            )
    aligned = _aligned_trace_center(traces, observation_stat=observation_stat)
    if aligned is None:
        return
    times, values = aligned
    (line,) = axis.plot(
        times,
        values,
        color=color,
        linestyle=linestyle,
        linewidth=1.9 if linestyle == "-" else 1.4,
        marker="o",
        markersize=2.8,
    )
    line.set_gid(gid)


def _aligned_trace_center(
    traces: list[pd.DataFrame],
    *,
    observation_stat: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not traces:
        return None
    times = traces[0]["time_from_event_h"].to_numpy(dtype=float)
    if any(not np.array_equal(times, trace["time_from_event_h"].to_numpy(dtype=float)) for trace in traces[1:]):
        return None
    values = np.vstack([trace["plot_value"].to_numpy(dtype=float) for trace in traces])
    center = np.mean(values, axis=0) if observation_stat == "mean" else np.median(values, axis=0)
    return times, center


def _draw_component_panel(axis: Any, *, diagnostic: ResponseWindowDiagnostic) -> None:
    y = np.arange(len(COMPONENT_COLUMNS))
    values = np.asarray(diagnostic.component_values)
    interval_low = np.asarray(diagnostic.component_descriptive_interval_low)
    interval_high = np.asarray(diagnostic.component_descriptive_interval_high)
    event_range = np.asarray(diagnostic.component_event_half_range)
    for index, (component, value) in enumerate(zip(COMPONENT_COLUMNS, values, strict=True)):
        state = component[1:]
        color = STATE_COLORS[state]
        axis.hlines(
            y[index],
            value - event_range[index],
            value + event_range[index],
            color=color,
            linewidth=5.0,
            alpha=0.20,
        )
        axis.hlines(y[index], interval_low[index], interval_high[index], color=color, linewidth=1.5)
        bound_kind = diagnostic.component_bound_kinds[index]
        axis.scatter(
            value,
            y[index],
            color=color,
            marker=BOUND_MARKERS[bound_kind],
            s=30.0,
            zorder=3,
        )
    axis.axvline(0.0, color="#64748b", linewidth=0.9)
    axis.axhline(3.5, color="#cbd5e1", linewidth=0.8)
    axis.set_yticks(y, labels=COMPONENT_COLUMNS)
    axis.invert_yaxis()
    axis.set_xlabel("Reduced value (log2 units)")
    notes = _quality_notes(diagnostic)
    if notes:
        axis.text(
            0.0,
            -0.16,
            "Quality flags — " + "; ".join(notes),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color="#7c2d12",
            wrap=True,
        )


def _quality_notes(diagnostic: ResponseWindowDiagnostic) -> list[str]:
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
        if diagnostic.component_event_has_policy_clipping[index]:
            flags.append("event-range clipping")
        if diagnostic.component_event_has_instrument_overflow[index]:
            flags.append("event-range overflow")
        if flags:
            notes.append(f"{component}: {', '.join(flags)}")
    return notes


def _has_quality_flags(diagnostic: ResponseWindowDiagnostic) -> bool:
    trace_flags = (
        diagnostic.traces["value_policy_clipped"].astype(bool)
        | diagnostic.traces["value_instrument_overflow"].astype(bool)
        | diagnostic.traces["value_bound_kind"].astype(str).ne("exact")
    ).any()
    return bool(trace_flags or _quality_notes(diagnostic))


__all__ = ["BOUND_MARKERS", "STATE_COLORS", "render_response_window_diagnostic"]
