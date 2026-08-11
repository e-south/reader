"""Render a provenance-aware four-state event-window diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from reader_workbench.plotting.style import use_style

from .diagnostic import FourStateEventWindowDiagnostic, prepare_four_state_event_window_diagnostic
from .diagnostic_components import draw_component_panel, has_quality_flags
from .diagnostic_style import BOUND_MARKERS, STATE_COLORS, STATE_MARKERS, validated_axis_labels, validated_state_labels
from .schema import STATE_ORDER


def render_four_state_event_window_diagnostic(
    traces: pd.DataFrame,
    designs: pd.DataFrame,
    *,
    source_experiment_id: str,
    design_id: str,
    reduction_id: str,
    pre_window_duration_h: float | None,
    axis_labels: Mapping[str, str] | None = None,
    state_labels: Mapping[str, str] | None = None,
    reference_label: str = "reference",
    title: str | None = None,
) -> Any:
    """Render observed traces, descriptive dispersion, and event sensitivity in one row."""

    diagnostic = prepare_four_state_event_window_diagnostic(
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
        figure.set_gid("four-state-event-window-diagnostic")
        labels = validated_axis_labels(axis_labels)
        states = validated_state_labels(state_labels)
        for axis, signal_kind, panel_title in (
            (axes[0], "growth", "Growth"),
            (axes[1], "response", "Reporter ratio"),
            (axes[2], "magnitude", "Signal vs reference"),
        ):
            _draw_trace_panel(axis, diagnostic=diagnostic, signal_kind=signal_kind)
            axis.set_title(panel_title, pad=8)
            axis.set_xlabel("Time from event estimate (h)")
            axis.set_ylabel(labels[signal_kind])
        draw_component_panel(
            axes[3],
            diagnostic=diagnostic,
            axis_labels=labels,
            reference_label=reference_label,
        )
        axes[3].set_title("Response-window phenotype", pad=8)

        interval_mass_percent = diagnostic.descriptive_interval_mass * 100.0
        response_basis = " · response shown as post − pre" if diagnostic.response_basis == "post_minus_pre" else ""
        metadata = (
            f"{diagnostic.window[0]:g}–{diagnostic.window[1]:g} h after "
            f"{diagnostic.event_id.replace('_', ' ')} · {diagnostic.observation_stat} across observations · "
            f"{interval_mass_percent:g}% resampling range · event timing ±{diagnostic.event_time_uncertainty_h:.2g} h"
            f"{response_basis}"
        )
        figure.suptitle(f"{title or f'{diagnostic.source_experiment_id} :: {diagnostic.design_id}'}\n{metadata}")

        legend = [
            Line2D(
                [0],
                [0],
                color=STATE_COLORS[state],
                marker=STATE_MARKERS[state],
                linewidth=1.8,
                label=states[state],
            )
            for state in STATE_ORDER
        ]
        if diagnostic.reference_design_id != diagnostic.design_id:
            legend.append(Line2D([0], [0], color="#9AA3AD", linestyle="--", linewidth=1.2, label=reference_label))
        legend.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="#64748b",
                    linewidth=5.0,
                    alpha=0.25,
                    label="event-time range",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#64748b",
                    linewidth=1.5,
                    label=f"{interval_mass_percent:g}% resampling range",
                ),
            ]
        )
        if has_quality_flags(diagnostic):
            legend.append(Line2D([0], [0], color="#7c2d12", marker="x", linestyle="none", label="quality/bound flag"))
        figure.legend(handles=legend, loc="outside lower center", ncol=min(len(legend), 8), frameon=False)
    return figure


def _draw_trace_panel(axis: Any, *, diagnostic: FourStateEventWindowDiagnostic, signal_kind: str) -> None:
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
            marker=STATE_MARKERS[state],
            linestyle="-",
            reference=False,
            observation_stat=diagnostic.observation_stat,
            gid="four-state-event-window-trace",
        )
        if signal_kind == "magnitude" and diagnostic.reference_design_id != diagnostic.design_id:
            reference = rows.loc[
                rows["design_id"].astype(str).eq(diagnostic.reference_design_id) & rows["state"].astype(str).eq(state)
            ]
            _draw_observed_traces(
                axis,
                reference,
                color="#9AA3AD",
                marker=STATE_MARKERS[state],
                linestyle="--",
                reference=True,
                observation_stat=diagnostic.observation_stat,
                gid="four-state-event-window-reference-trace",
            )


def _draw_time_context(axis: Any, *, diagnostic: FourStateEventWindowDiagnostic, signal_kind: str) -> None:
    uncertainty = diagnostic.event_time_uncertainty_h
    if uncertainty > 0.0:
        interval = axis.axvspan(-uncertainty, uncertainty, color="#64748b", alpha=0.10, linewidth=0.0)
        interval.set_gid("four-state-event-window-event-interval")
    event = axis.axvline(0.0, color="#64748b", linewidth=0.9)
    event.set_gid("four-state-event-window-event-estimate")
    post = axis.axvspan(*diagnostic.window, color="#f0c36e", alpha=0.18, linewidth=0.0)
    post.set_gid("four-state-event-window-reduction-window")
    if signal_kind == "response" and diagnostic.pre_window is not None:
        pre = axis.axvspan(*diagnostic.pre_window, color="#94a3b8", alpha=0.14, linewidth=0.0)
        pre.set_gid("four-state-event-window-pre-window")


def _draw_observed_traces(
    axis: Any,
    rows: pd.DataFrame,
    *,
    color: str,
    marker: str,
    linestyle: str,
    reference: bool,
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
            linewidth=0.65 if reference else 0.8,
            marker=marker,
            markersize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.3,
            alpha=0.10 if reference else 0.15,
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
        marker=marker,
        markersize=4.0 if not reference else 3.4,
        markeredgecolor="white",
        markeredgewidth=0.55 if not reference else 0.45,
        alpha=0.55 if reference else 1.0,
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


__all__ = ["BOUND_MARKERS", "STATE_COLORS", "STATE_MARKERS", "render_four_state_event_window_diagnostic"]
