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

_REFERENCE_COLOR = "#7C8793"


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
            "figure_figsize": (18.5, 6.0),
            "font_size": 13.0,
            "axes_labelsize": 13.0,
            "axes_titlesize": 15.0,
            "xtick_labelsize": 11.5,
            "ytick_labelsize": 11.5,
            "legend_fontsize": 11.5,
        }
    ):
        from matplotlib.figure import Figure  # noqa: PLC0415
        from matplotlib.lines import Line2D  # noqa: PLC0415

        figure = Figure(
            figsize=(18.5, 6.0),
            constrained_layout=True,
        )
        axes = figure.subplots(
            1,
            4,
            gridspec_kw={"width_ratios": (1.0, 1.0, 1.0, 0.95)},
        )
        figure.set_gid("four-state-event-window-diagnostic")
        labels = validated_axis_labels(axis_labels)
        states = validated_state_labels(state_labels)
        for axis, signal_kind, panel_title in (
            (axes[0], "growth", "Growth"),
            (axes[1], "response", "Response"),
            (axes[2], "magnitude", "Signal vs reference"),
        ):
            _draw_trace_panel(axis, diagnostic=diagnostic, signal_kind=signal_kind)
            axis.set_title(panel_title, pad=12)
            axis.set_xlabel("Time from event estimate (h)", labelpad=8)
            axis.set_ylabel(labels[signal_kind], labelpad=8)
        draw_component_panel(
            axes[3],
            diagnostic=diagnostic,
            axis_labels=labels,
            reference_label=reference_label,
        )
        window_start, window_end = diagnostic.window
        axes[3].set_title(f"{window_start:g}–{window_end:g} h mean", pad=12)
        for axis in axes:
            axis.set_box_aspect(1.0)

        figure.suptitle(
            title or f"{diagnostic.source_experiment_id} :: {diagnostic.design_id}",
            fontsize=19.0,
            fontweight="bold",
        )

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
            legend.append(
                Line2D([0], [0], color=_REFERENCE_COLOR, linestyle="--", linewidth=1.7, label=reference_label)
            )
        if has_quality_flags(diagnostic):
            legend.append(Line2D([0], [0], color="#7c2d12", marker="x", linestyle="none", label="quality/bound flag"))
        figure.legend(
            handles=legend,
            loc="outside lower center",
            ncol=min(len(legend), 8),
            frameon=False,
            columnspacing=1.5,
            handletextpad=0.55,
        )
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
            interval_mass=diagnostic.descriptive_interval_mass,
            gid="four-state-event-window-trace",
        )
        if signal_kind == "magnitude" and diagnostic.reference_design_id != diagnostic.design_id:
            reference = rows.loc[
                rows["design_id"].astype(str).eq(diagnostic.reference_design_id) & rows["state"].astype(str).eq(state)
            ]
            _draw_observed_traces(
                axis,
                reference,
                color=_REFERENCE_COLOR,
                marker=STATE_MARKERS[state],
                linestyle="--",
                reference=True,
                observation_stat=diagnostic.observation_stat,
                interval_mass=diagnostic.descriptive_interval_mass,
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
    interval_mass: float,
    gid: str,
) -> None:
    traces: list[pd.DataFrame] = []
    for _, trace in rows.groupby("position", sort=True):
        trace = trace.sort_values("time_from_event_h", kind="stable")
        traces.append(trace)
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
    summary = summarize_observed_traces(
        traces,
        observation_stat=observation_stat,
        interval_mass=interval_mass,
    )
    if summary is None:
        raise ValueError("four-state event-window diagnostic requires observed trace points")
    times, values, interval_low, interval_high = summary
    band = axis.fill_between(
        times,
        interval_low,
        interval_high,
        color=color,
        alpha=0.07 if reference else 0.14,
        linewidth=0.0,
        zorder=1,
    )
    band.set_gid("four-state-event-window-observation-interval")
    (line,) = axis.plot(
        times,
        values,
        color=color,
        linestyle=linestyle,
        linewidth=1.9 if not reference else 1.15,
        marker=marker,
        markersize=4.0 if not reference else 2.7,
        markeredgecolor="white",
        markeredgewidth=0.55 if not reference else 0.45,
        alpha=0.78 if reference else 1.0,
        markevery=4 if reference else None,
        zorder=3,
    )
    line.set_gid(gid)


def summarize_observed_traces(
    traces: list[pd.DataFrame],
    *,
    observation_stat: str,
    interval_mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not traces:
        return None
    observed = pd.concat(
        (
            trace.loc[:, ["time_from_event_h", "plot_value"]].assign(trace_index=index)
            for index, trace in enumerate(traces)
        ),
        ignore_index=True,
    )
    grouped = observed.groupby("time_from_event_h", sort=True, observed=True)["plot_value"]
    times = grouped.size().index.to_numpy(dtype=float)
    center = (
        grouped.mean().to_numpy(dtype=float) if observation_stat == "mean" else grouped.median().to_numpy(dtype=float)
    )
    tail = (1.0 - interval_mass) / 2.0
    interval_low = grouped.quantile(tail).to_numpy(dtype=float)
    interval_high = grouped.quantile(1.0 - tail).to_numpy(dtype=float)
    return times, center, interval_low, interval_high


__all__ = [
    "BOUND_MARKERS",
    "STATE_COLORS",
    "STATE_MARKERS",
    "render_four_state_event_window_diagnostic",
    "summarize_observed_traces",
]
