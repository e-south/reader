from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reader.domains.plate_reader.plots import _retron_sponge_presentation as retron_presentation
from reader.domains.plate_reader.plots.common import annotate_points_smart, best_subplot_grid, shared_numeric_limits

_FAMILY_COLOR_MAP = {
    "mono": "#0072B2",
    "bi": "#E69F00",
    "tri": "#009E73",
    "quad": "#CC79A7",
    "control": "#6f6f6f",
    "other": "#56B4E9",
}


@dataclass(frozen=True)
class _FingerprintFigurePayload:
    selected_sponge: str
    sensor_levels: tuple[str, ...]
    comparison_order: tuple[str, ...]
    stats: pd.DataFrame
    y_limits: tuple[float, float]
    max_sources: int
    width: float
    offsets: dict[str, float]
    x_positions: dict[str, float]
    comparison_colors: dict[str, str]
    edge_colors: dict[str, str]
    point_facecolors: dict[str, str]


@dataclass(frozen=True)
class _AggregateParetoFigurePayload:
    family_levels: tuple[str, ...]
    color_map: dict[str, Any]
    sizes: pd.Series
    annotation_points: tuple[tuple[float, float], ...]
    annotation_labels: tuple[str, ...]


@dataclass(frozen=True)
class _SensorScatterFigurePolicy:
    height: float
    point_size: float
    annotation_fontsize: float = 7.2
    annotation_max_parts_per_line: int = 2
    figure_width_per_sensor: float = 4.3
    left: float = 0.09
    right: float = 0.88
    top: float = 0.88
    bottom: float = 0.16
    wspace: float = 0.12
    legend_anchor_x: float = 0.94
    add_horizontal_zero: bool = False
    add_identity_line: bool = False
    equal_aspect: bool = False


@dataclass(frozen=True)
class _SingleAxisAxisPolicy:
    xlabel: str
    ylabel: str
    x_tick_size: float
    y_tick_size: float
    grid_axis: str | None = None
    grid_color: str = "#d9d9d9"
    grid_linewidth: float = 0.6
    grid_alpha: float = 0.50


@dataclass(frozen=True)
class _SingleAxisFigurePolicy:
    figsize: tuple[float, float]
    suptitle: str | None = None
    suptitle_y: float | None = None
    subtitle: str | None = None
    subtitle_y: float | None = None
    left: float = 0.11
    right: float = 0.80
    top: float = 0.84
    bottom: float = 0.16


@dataclass(frozen=True)
class _SingleAxisLegendPolicy:
    loc: str
    bbox_to_anchor: tuple[float, float]
    borderaxespad: float = 0.0
    frameon: bool = False
    title: str | None = None


@dataclass(frozen=True)
class _MatrixHeatmapFigurePolicy:
    min_width: float
    width_padding: float
    width_per_column: float
    min_height: float
    height_padding: float
    height_per_row: float
    cmap: str
    center: float
    annotation_format: str
    annotation_fontsize: float
    linewidths: float
    linecolor: str
    cbar_label: str
    cbar_shrink: float
    title: str
    title_pad: float
    title_fontsize: float
    xlabel: str
    xlabel_fontsize: float
    xtick_labelsize: float
    ytick_labelsize: float
    wrap_xtick_parts_per_line: int


def build_specificity_matrix_figure(*, matrix: pd.DataFrame, score_metric: str) -> Any | None:
    if matrix.empty:
        return None
    policy = _specificity_matrix_figure_policy(score_metric)
    figure, axis = _new_matrix_heatmap_figure(matrix=matrix, policy=policy)
    _plot_matrix_heatmap(axis, matrix=matrix, policy=policy)
    _decorate_matrix_heatmap_axis(axis, policy=policy)
    return figure


def build_aggregate_pareto_figure(
    *,
    pareto_df: pd.DataFrame,
    score_metric: str,
    burden_metric: str,
) -> Any | None:
    if pareto_df.empty:
        return None
    payload = _aggregate_pareto_figure_payload(pareto_df)
    figure_policy = _aggregate_pareto_figure_policy()
    axis_policy = _aggregate_pareto_axis_policy(score_metric=score_metric, burden_metric=burden_metric)
    figure, axis = _new_single_axis_figure(policy=figure_policy)
    _plot_aggregate_pareto_points(axis, pareto_df=pareto_df, payload=payload)
    _decorate_aggregate_pareto_axis(axis, payload=payload, policy=axis_policy)
    legend_handles = _aggregate_pareto_legend_handles(payload)
    _apply_single_axis_legend(
        axis,
        policy=_aggregate_pareto_legend_policy(),
        handles=legend_handles,
    )
    _finalize_single_axis_figure(figure, policy=figure_policy)
    return figure


def build_architecture_figure(
    *,
    architecture_df: pd.DataFrame,
    score_metric: str,
    architecture_x: str,
) -> Any | None:
    if architecture_df.empty:
        return None

    sensors = sorted(architecture_df["sensor"].dropna().astype(str).unique())
    palette = _family_palette(architecture_df)
    x_limits = shared_numeric_limits(
        architecture_df[architecture_x].to_numpy(dtype=float, copy=False),
        pad_fraction=0.10,
        min_span=1.0,
    )
    y_limits = shared_numeric_limits(
        architecture_df["value"].to_numpy(dtype=float, copy=False),
        center=0.0,
        pad_fraction=0.10,
        min_span=0.10,
    )
    return _build_sensor_scatter_figure(
        frame=architecture_df,
        sensors=sensors,
        palette=palette,
        x_column=architecture_x,
        y_column="value",
        x_limits=x_limits,
        y_limits=y_limits,
        policy=_architecture_sensor_scatter_policy(),
        xlabel=_architecture_axis_label(architecture_x),
        ylabel=_aggregate_score_axis_label(score_metric),
    )


def build_expected_vs_observed_figure(
    *,
    expected_vs_observed_df: pd.DataFrame,
    score_metric: str,
    expected_mode: str,
) -> Any | None:
    if expected_vs_observed_df.empty:
        return None

    sensors = sorted(expected_vs_observed_df["sensor"].dropna().astype(str).unique())
    palette = _family_palette(expected_vs_observed_df)
    combined_limits = shared_numeric_limits(
        pd.concat(
            [
                expected_vs_observed_df[expected_mode],
                expected_vs_observed_df["observed"],
            ],
            ignore_index=True,
        ).to_numpy(dtype=float, copy=False),
        center=0.0,
        pad_fraction=0.10,
        min_span=0.10,
    )
    return _build_sensor_scatter_figure(
        frame=expected_vs_observed_df,
        sensors=sensors,
        palette=palette,
        x_column=expected_mode,
        y_column="observed",
        x_limits=combined_limits,
        y_limits=combined_limits,
        policy=_expected_vs_observed_sensor_scatter_policy(),
        xlabel=_expected_axis_label(expected_mode, score_metric=score_metric),
        ylabel=f"Observed multifunction score ({_aggregate_score_axis_label(score_metric)})",
    )


def build_fingerprint_figure(
    *,
    fingerprint_df: pd.DataFrame,
    score_metric: str,
) -> Any | None:
    if fingerprint_df.empty:
        return None
    selected_sponges = sorted(
        fingerprint_df["selected_sponge"].dropna().astype(str).unique().tolist(),
        key=str,
    )
    global_limits = shared_numeric_limits(
        fingerprint_df["value"].to_numpy(dtype=float, copy=False),
        center=0.0,
        pad_fraction=0.12,
        min_span=0.10,
    )
    payloads = [
        _fingerprint_figure_payload(
            fingerprint_df[fingerprint_df["selected_sponge"].astype(str) == selected_sponge].copy(),
            y_limits=global_limits,
        )
        for selected_sponge in selected_sponges
    ]
    axis_policy = _fingerprint_axis_policy(score_metric)
    if len(payloads) == 1:
        payload = payloads[0]
        figure_policy = _fingerprint_figure_policy(payload=payload, subtitle=_fingerprint_support_text(fingerprint_df))
        figure, axis = _new_single_axis_figure(policy=figure_policy)
        _plot_fingerprint_subplot(axis=axis, payload=payload, subplot_df=fingerprint_df)
        _decorate_fingerprint_axis(axis, payload=payload, policy=axis_policy)
        _apply_single_axis_legend(axis, policy=_fingerprint_legend_policy())
        _finalize_single_axis_figure(figure, policy=figure_policy)
        return figure
    return _build_fingerprint_grid_figure(
        fingerprint_df=fingerprint_df,
        payloads=payloads,
        axis_policy=axis_policy,
    )


def _aggregate_pareto_figure_payload(pareto_df: pd.DataFrame) -> _AggregateParetoFigurePayload:
    family_levels = tuple(_ordered_text(pareto_df["sponge_family_size"].fillna("other").astype(str).tolist()))
    color_values = sns.color_palette("colorblind", n_colors=max(1, len(family_levels)))
    color_map = {family: color_values[idx % len(color_values)] for idx, family in enumerate(family_levels)}
    sizes = 90.0 + 260.0 * pareto_df["leakiness"].fillna(0.0)
    return _AggregateParetoFigurePayload(
        family_levels=family_levels,
        color_map=color_map,
        sizes=sizes,
        annotation_points=tuple((float(row["on_target"]), float(row["burden"])) for _, row in pareto_df.iterrows()),
        annotation_labels=tuple(
            _wrap_hyphenated_plot_label(str(row["sponge"]), max_parts_per_line=2) for _, row in pareto_df.iterrows()
        ),
    )


def _plot_aggregate_pareto_points(
    axis: Any,
    *,
    pareto_df: pd.DataFrame,
    payload: _AggregateParetoFigurePayload,
) -> None:
    axis.scatter(
        pareto_df["on_target"],
        pareto_df["burden"],
        s=payload.sizes,
        c=[payload.color_map.get(str(item), "#4c72b0") for item in pareto_df["sponge_family_size"].fillna("other")],
        alpha=0.85,
        edgecolors="#222222",
        linewidths=0.5,
        zorder=2,
    )


def _aggregate_pareto_legend_handles(payload: _AggregateParetoFigurePayload) -> list[Any]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=family,
            markerfacecolor=payload.color_map[family],
            markeredgecolor="#222222",
            markersize=7,
        )
        for family in payload.family_levels
    ]


def _fingerprint_figure_payload(
    fingerprint_df: pd.DataFrame,
    *,
    y_limits: tuple[float, float] | None = None,
) -> _FingerprintFigurePayload:
    sensor_levels = tuple(sorted(fingerprint_df["sensor"].dropna().astype(str).unique().tolist()))
    comparison_order = tuple(_fingerprint_comparison_order(fingerprint_df))
    comparison_colors, edge_colors, point_facecolors = _fingerprint_comparison_styles()
    source_counts = (
        fingerprint_df.groupby(["sensor", "comparison_group"], dropna=False)["value"]
        .size()
        .reset_index(name="n_sources")
    )
    width = 0.34
    offsets = _group_offsets(comparison_order, width=width)
    return _FingerprintFigurePayload(
        selected_sponge=str(fingerprint_df["selected_sponge"].dropna().astype(str).iloc[0]),
        sensor_levels=sensor_levels,
        comparison_order=comparison_order,
        stats=_fingerprint_group_stats(fingerprint_df),
        y_limits=y_limits
        or shared_numeric_limits(
            fingerprint_df["value"].to_numpy(dtype=float, copy=False),
            center=0.0,
            pad_fraction=0.12,
            min_span=0.10,
        ),
        max_sources=int(source_counts["n_sources"].max()) if not source_counts.empty else 0,
        width=width,
        offsets=offsets,
        x_positions={sensor: float(idx) for idx, sensor in enumerate(sensor_levels)},
        comparison_colors=comparison_colors,
        edge_colors=edge_colors,
        point_facecolors=point_facecolors,
    )


def _new_single_axis_figure(*, policy: _SingleAxisFigurePolicy) -> tuple[Any, Any]:
    return plt.subplots(figsize=policy.figsize, constrained_layout=False)


def _decorate_aggregate_pareto_axis(
    axis: Any,
    *,
    payload: _AggregateParetoFigurePayload,
    policy: _SingleAxisAxisPolicy,
) -> None:
    _annotate_single_axis_points(
        axis,
        points=payload.annotation_points,
        labels=payload.annotation_labels,
        fontsize=8.0,
    )
    axis.axvline(0.0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    _style_single_axis(axis, policy=policy)
    with suppress(Exception):
        axis.set_box_aspect(1.0)


def _decorate_fingerprint_axis(
    axis: Any,
    *,
    payload: _FingerprintFigurePayload,
    policy: _SingleAxisAxisPolicy,
) -> None:
    axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    axis.set_xlim(-0.55, max(0.55, len(payload.sensor_levels) - 0.45))
    axis.set_ylim(payload.y_limits)
    axis.set_xticks([payload.x_positions[sensor] for sensor in payload.sensor_levels])
    axis.set_xticklabels(payload.sensor_levels)
    _style_single_axis(axis, policy=policy)


def _plot_fingerprint_subplot(
    *,
    axis: Any,
    payload: _FingerprintFigurePayload,
    subplot_df: pd.DataFrame,
) -> None:
    _plot_fingerprint_bars(
        axis,
        stats=payload.stats,
        sensor_levels=payload.sensor_levels,
        comparison_order=payload.comparison_order,
        x_positions=payload.x_positions,
        offsets=payload.offsets,
        width=payload.width,
        comparison_colors=payload.comparison_colors,
        edge_colors=payload.edge_colors,
    )
    _plot_fingerprint_points(
        axis,
        fingerprint_df=subplot_df,
        sensor_levels=payload.sensor_levels,
        comparison_order=payload.comparison_order,
        x_positions=payload.x_positions,
        offsets=payload.offsets,
        point_facecolors=payload.point_facecolors,
        edge_colors=payload.edge_colors,
    )


def _new_matrix_heatmap_figure(*, matrix: pd.DataFrame, policy: _MatrixHeatmapFigurePolicy) -> tuple[Any, Any]:
    return plt.subplots(
        figsize=_matrix_heatmap_figure_size(matrix=matrix, policy=policy),
        constrained_layout=True,
    )


def _style_single_axis(axis: Any, *, policy: _SingleAxisAxisPolicy) -> None:
    axis.set_xlabel(policy.xlabel, fontsize=11)
    axis.set_ylabel(policy.ylabel, fontsize=11)
    axis.tick_params(axis="x", labelsize=policy.x_tick_size)
    axis.tick_params(axis="y", labelsize=policy.y_tick_size)
    if policy.grid_axis:
        axis.grid(
            axis=policy.grid_axis,
            color=policy.grid_color,
            linewidth=policy.grid_linewidth,
            alpha=policy.grid_alpha,
        )


def _annotate_single_axis_points(
    axis: Any,
    *,
    points: Sequence[tuple[float, float]],
    labels: Sequence[str],
    fontsize: float,
) -> None:
    if not points:
        return
    annotate_points_smart(
        ax=axis,
        points=list(points),
        labels=list(labels),
        text_kwargs={"fontsize": fontsize},
    )


def _apply_single_axis_legend(
    axis: Any,
    *,
    policy: _SingleAxisLegendPolicy,
    handles: Sequence[Any] | None = None,
) -> None:
    if handles is not None and not handles:
        return
    legend_kwargs = {
        "frameon": policy.frameon,
        "title": policy.title,
        "loc": policy.loc,
        "bbox_to_anchor": policy.bbox_to_anchor,
        "borderaxespad": policy.borderaxespad,
    }
    if handles is None:
        axis.legend(**legend_kwargs)
        return
    axis.legend(handles=list(handles), **legend_kwargs)


def _finalize_single_axis_figure(figure: Any, *, policy: _SingleAxisFigurePolicy) -> None:
    if policy.suptitle:
        figure.suptitle(
            policy.suptitle,
            y=policy.suptitle_y or 0.97,
            x=0.5,
            ha="center",
            fontweight="normal",
            fontsize=13,
        )
    if policy.subtitle:
        figure.text(
            0.5,
            policy.subtitle_y or 0.92,
            policy.subtitle,
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
        )
    figure.subplots_adjust(bottom=policy.bottom, left=policy.left, right=policy.right, top=policy.top)


def _matrix_heatmap_figure_size(*, matrix: pd.DataFrame, policy: _MatrixHeatmapFigurePolicy) -> tuple[float, float]:
    n_rows = max(1, len(matrix.index))
    n_cols = max(1, len(matrix.columns))
    return (
        max(policy.min_width, policy.width_padding + policy.width_per_column * n_cols),
        max(policy.min_height, policy.height_padding + policy.height_per_row * n_rows),
    )


def _plot_matrix_heatmap(axis: Any, *, matrix: pd.DataFrame, policy: _MatrixHeatmapFigurePolicy) -> None:
    sns.heatmap(
        matrix,
        ax=axis,
        cmap=policy.cmap,
        center=policy.center,
        annot=True,
        fmt=policy.annotation_format,
        annot_kws={"fontsize": policy.annotation_fontsize},
        cbar=True,
        square=True,
        linewidths=policy.linewidths,
        linecolor=policy.linecolor,
        cbar_kws={"label": policy.cbar_label, "shrink": policy.cbar_shrink},
    )


def _decorate_matrix_heatmap_axis(axis: Any, *, policy: _MatrixHeatmapFigurePolicy) -> None:
    axis.set_title(policy.title, pad=policy.title_pad, fontweight="normal", fontsize=policy.title_fontsize)
    axis.set_xlabel(policy.xlabel, fontsize=policy.xlabel_fontsize)
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelrotation=0, labelsize=policy.xtick_labelsize)
    axis.tick_params(axis="y", labelrotation=0, labelsize=policy.ytick_labelsize)
    axis.set_xticklabels(
        [
            _wrap_hyphenated_plot_label(label.get_text(), max_parts_per_line=policy.wrap_xtick_parts_per_line)
            for label in axis.get_xticklabels()
        ]
    )
    for label in axis.get_xticklabels():
        label.set_ha("center")


def _specificity_matrix_figure_policy(score_metric: str) -> _MatrixHeatmapFigurePolicy:
    return _MatrixHeatmapFigurePolicy(
        min_width=7.4,
        width_padding=2.2,
        width_per_column=0.72,
        min_height=3.2,
        height_padding=1.6,
        height_per_row=0.72,
        cmap="vlag",
        center=0.0,
        annotation_format=".2f",
        annotation_fontsize=8,
        linewidths=0.3,
        linecolor="#f0f0f0",
        cbar_label=_metric_axis_label(score_metric),
        cbar_shrink=0.80,
        title="Relevant-stress target activity matrix",
        title_pad=10,
        title_fontsize=11,
        xlabel="Sponge design",
        xlabel_fontsize=10,
        xtick_labelsize=8,
        ytick_labelsize=9,
        wrap_xtick_parts_per_line=2,
    )


def _aggregate_pareto_axis_policy(*, score_metric: str, burden_metric: str) -> _SingleAxisAxisPolicy:
    return _SingleAxisAxisPolicy(
        xlabel=f"Mean {_aggregate_score_axis_label(score_metric)}",
        ylabel="Mean burden penalty",
        x_tick_size=7.0,
        y_tick_size=7.0,
    )


def _aggregate_pareto_figure_policy() -> _SingleAxisFigurePolicy:
    return _SingleAxisFigurePolicy(
        figsize=(6.4, 4.8),
        suptitle="Aggregate pareto ranking",
        suptitle_y=0.97,
        subtitle="Absolute expected-direction effect versus burden penalty across the review set; point size encodes |L_pre|.",
        subtitle_y=0.93,
        bottom=0.12,
        left=0.12,
        right=0.78,
        top=0.85,
    )


def _aggregate_pareto_legend_policy() -> _SingleAxisLegendPolicy:
    return _SingleAxisLegendPolicy(loc="center left", bbox_to_anchor=(1.01, 0.5))


def _fingerprint_axis_policy(score_metric: str) -> _SingleAxisAxisPolicy:
    return _SingleAxisAxisPolicy(
        xlabel="Relevant sensor arm",
        ylabel=_metric_axis_label(score_metric),
        x_tick_size=9.6,
        y_tick_size=8.8,
        grid_axis="y",
    )


def _fingerprint_figure_policy(*, payload: _FingerprintFigurePayload, subtitle: str) -> _SingleAxisFigurePolicy:
    return _SingleAxisFigurePolicy(
        figsize=(max(6.2, 2.0 + 1.42 * len(payload.sensor_levels)), 4.45),
        suptitle=f"Relevant sensor arms · {payload.selected_sponge}",
        suptitle_y=0.97,
        subtitle=subtitle,
        subtitle_y=0.925,
        left=0.10,
        right=0.86,
        top=0.82,
        bottom=0.17,
    )


def _fingerprint_legend_policy() -> _SingleAxisLegendPolicy:
    return _SingleAxisLegendPolicy(loc="upper left", bbox_to_anchor=(1.00, 1.0))


def _build_fingerprint_grid_figure(
    *,
    fingerprint_df: pd.DataFrame,
    payloads: Sequence[_FingerprintFigurePayload],
    axis_policy: _SingleAxisAxisPolicy,
) -> Any:
    rows, cols = best_subplot_grid(len(payloads))
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(max(7.6, 3.8 * cols), max(4.1, 3.45 * rows)),
        constrained_layout=False,
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.ravel()
    for axis, payload in zip(axes_flat, payloads, strict=False):
        subplot_df = fingerprint_df[fingerprint_df["selected_sponge"].astype(str) == payload.selected_sponge].copy()
        _plot_fingerprint_subplot(axis=axis, payload=payload, subplot_df=subplot_df)
        _decorate_fingerprint_axis(axis, payload=payload, policy=axis_policy)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(str(payload.selected_sponge), pad=6, fontsize=10, fontweight="normal")
    for axis in axes_flat[len(payloads) :]:
        axis.set_visible(False)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=group,
            markerfacecolor=payloads[0].comparison_colors.get(group, "#4c72b0"),
            markeredgecolor=payloads[0].edge_colors.get(group, "#222222"),
            markersize=7,
        )
        for group in payloads[0].comparison_order
    ]
    figure.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.93, 0.5),
        borderaxespad=0.0,
        title=None,
    )
    figure.suptitle("Relevant sensor arms by sponge", y=0.975, fontsize=13, fontweight="normal")
    figure.text(
        0.5,
        0.935,
        _fingerprint_support_text(fingerprint_df),
        ha="center",
        va="top",
        fontsize=9.4,
        color="#333333",
    )
    figure.supxlabel(axis_policy.xlabel, y=0.08)
    figure.supylabel(axis_policy.ylabel, x=0.03)
    figure.subplots_adjust(left=0.10, right=0.88, top=0.84, bottom=0.17, hspace=0.30, wspace=0.20)
    return figure


def _architecture_sensor_scatter_policy() -> _SensorScatterFigurePolicy:
    return _SensorScatterFigurePolicy(
        height=4.55,
        point_size=82,
        annotation_fontsize=7.1,
        right=0.89,
        legend_anchor_x=0.93,
        add_horizontal_zero=True,
    )


def _expected_vs_observed_sensor_scatter_policy() -> _SensorScatterFigurePolicy:
    return _SensorScatterFigurePolicy(
        height=4.65,
        point_size=84,
        annotation_fontsize=7.1,
        right=0.89,
        legend_anchor_x=0.93,
        add_identity_line=True,
        equal_aspect=True,
    )


def _family_palette(frame: pd.DataFrame) -> dict[str, str]:
    family_levels = _ordered_text(frame["sponge_family_size"].fillna("other").astype(str).tolist())
    return {level: _FAMILY_COLOR_MAP.get(level, _FAMILY_COLOR_MAP["other"]) for level in family_levels}


def _sensor_subplot_figure(
    *,
    sensors: Sequence[str],
    policy: _SensorScatterFigurePolicy,
) -> tuple[Any, Any]:
    return plt.subplots(
        1,
        len(sensors),
        figsize=(policy.figure_width_per_sensor * len(sensors), policy.height),
        constrained_layout=False,
        squeeze=False,
        sharex=True,
        sharey=True,
    )


def _build_sensor_scatter_figure(
    *,
    frame: pd.DataFrame,
    sensors: Sequence[str],
    palette: Mapping[str, str],
    x_column: str,
    y_column: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    policy: _SensorScatterFigurePolicy,
    xlabel: str,
    ylabel: str,
) -> Any:
    figure, axes = _sensor_subplot_figure(sensors=sensors, policy=policy)
    for axis, sensor in zip(axes[0], sensors, strict=True):
        sensor_df = frame[frame["sensor"].astype(str) == sensor].copy()
        _plot_sensor_family_scatter(
            axis,
            sensor_df=sensor_df,
            x_column=x_column,
            y_column=y_column,
            palette=palette,
            point_size=policy.point_size,
            annotation_fontsize=policy.annotation_fontsize,
            annotation_max_parts_per_line=policy.annotation_max_parts_per_line,
        )
        _decorate_sensor_scatter_axis(
            axis,
            sensor=sensor,
            x_limits=x_limits,
            y_limits=y_limits,
            policy=policy,
        )
    _finalize_sensor_scatter_figure(figure, axes[0], xlabel=xlabel, ylabel=ylabel, policy=policy)
    return figure


def _decorate_sensor_scatter_axis(
    axis: Any,
    *,
    sensor: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    policy: _SensorScatterFigurePolicy,
) -> None:
    if policy.add_horizontal_zero:
        axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0)
    if policy.add_identity_line:
        axis.plot(
            [x_limits[0], x_limits[1]],
            [y_limits[0], y_limits[1]],
            color="#777777",
            linestyle=":",
            linewidth=1.0,
        )
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    if policy.equal_aspect:
        axis.set_aspect("equal", adjustable="box")
    axis.set_title(str(sensor), pad=8, fontweight="normal")
    axis.set_xlabel("")
    axis.set_ylabel("")


def _plot_sensor_family_scatter(
    axis: Any,
    *,
    sensor_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    palette: Mapping[str, str],
    point_size: float,
    annotation_fontsize: float,
    annotation_max_parts_per_line: int,
) -> None:
    sns.scatterplot(
        data=sensor_df,
        x=x_column,
        y=y_column,
        hue="sponge_family_size",
        palette=palette,
        s=point_size,
        edgecolor="black",
        linewidth=0.45,
        ax=axis,
    )
    annotate_points_smart(
        ax=axis,
        points=[(float(row[x_column]), float(row[y_column])) for _, row in sensor_df.iterrows()],
        labels=[
            _wrap_hyphenated_plot_label(
                str(row["sponge"]),
                max_parts_per_line=annotation_max_parts_per_line,
            )
            for _, row in sensor_df.iterrows()
        ],
        text_kwargs={"fontsize": annotation_fontsize},
    )


def _finalize_sensor_scatter_figure(
    figure: Any,
    axes: Sequence[Any],
    *,
    xlabel: str,
    ylabel: str,
    policy: _SensorScatterFigurePolicy,
) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(policy.legend_anchor_x, 0.5),
            ncol=1,
            title=None,
        )
    for axis in axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
        with suppress(Exception):
            axis.set_box_aspect(1.0)
    figure.supxlabel(xlabel, y=0.09)
    figure.supylabel(ylabel, x=0.02)
    figure.subplots_adjust(
        bottom=policy.bottom,
        left=policy.left,
        right=policy.right,
        top=policy.top,
        wspace=policy.wspace,
    )


def _fingerprint_comparison_order(fingerprint_df: pd.DataFrame) -> list[str]:
    available = set(fingerprint_df["comparison_group"].astype(str))
    return [group for group in ("tetO reference", "Selected sponge") if group in available]


def _fingerprint_group_stats(fingerprint_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fingerprint_df.groupby(["sensor", "comparison_group"], dropna=False)["value"]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )


def _fingerprint_comparison_styles() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    return (
        {
            "tetO reference": "#f3ebe7",
            "Selected sponge": "#4c72b0",
        },
        {
            "tetO reference": "#9b7d72",
            "Selected sponge": "#1f3552",
        },
        {
            "tetO reference": "#ffffff",
            "Selected sponge": "#4c72b0",
        },
    )


def _group_offsets(groups: Sequence[str], *, width: float) -> dict[str, float]:
    return {group: ((idx - (len(groups) - 1) / 2.0) * width) for idx, group in enumerate(groups)}


def _plot_fingerprint_bars(
    axis: Any,
    *,
    stats: pd.DataFrame,
    sensor_levels: Sequence[str],
    comparison_order: Sequence[str],
    x_positions: Mapping[str, float],
    offsets: Mapping[str, float],
    width: float,
    comparison_colors: Mapping[str, str],
    edge_colors: Mapping[str, str],
) -> None:
    for group in comparison_order:
        group_stats = stats[stats["comparison_group"].astype(str) == group].copy()
        group_stats = group_stats.set_index("sensor").reindex(sensor_levels).reset_index()
        x_values = [x_positions[str(sensor)] + offsets[group] for sensor in group_stats["sensor"].astype(str)]
        mean_values = pd.to_numeric(group_stats["mean"], errors="coerce").to_numpy(dtype=float)
        error_values = pd.to_numeric(group_stats["sd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        axis.bar(
            x_values,
            mean_values,
            width=width * 0.92,
            color=comparison_colors.get(group, "#4c72b0"),
            edgecolor=edge_colors.get(group, "#222222"),
            linewidth=0.9,
            label=group,
            zorder=2,
        )
        replicate_counts = pd.to_numeric(group_stats["n"], errors="coerce").fillna(0).to_numpy(dtype=int)
        error_mask = (replicate_counts > 1) & np.isfinite(error_values)
        if error_mask.any():
            axis.errorbar(
                np.asarray(x_values, dtype=float)[error_mask],
                mean_values[error_mask],
                yerr=error_values[error_mask],
                fmt="none",
                ecolor=edge_colors.get(group, "#222222"),
                elinewidth=1.0,
                capsize=3.0,
                capthick=1.0,
                zorder=3,
            )


def _plot_fingerprint_points(
    axis: Any,
    *,
    fingerprint_df: pd.DataFrame,
    sensor_levels: Sequence[str],
    comparison_order: Sequence[str],
    x_positions: Mapping[str, float],
    offsets: Mapping[str, float],
    point_facecolors: Mapping[str, str],
    edge_colors: Mapping[str, str],
) -> None:
    for group in comparison_order:
        group_points = fingerprint_df[fingerprint_df["comparison_group"].astype(str) == group].copy()
        for sensor in sensor_levels:
            sensor_points = group_points[group_points["sensor"].astype(str) == sensor].copy()
            if sensor_points.empty:
                continue
            count = len(sensor_points)
            jitters = [0.0] if count == 1 else np.linspace(-0.06, 0.06, count).tolist()
            x_center = x_positions[str(sensor)] + offsets[group]
            for jitter, (_, point_row) in zip(jitters, sensor_points.iterrows(), strict=False):
                axis.scatter(
                    x_center + float(jitter),
                    float(point_row["value"]),
                    s=28,
                    facecolor=point_facecolors.get(group, "#4c72b0"),
                    edgecolor=edge_colors.get(group, "#222222"),
                    linewidth=0.8,
                    zorder=4,
                )


def _fingerprint_support_text(fingerprint_df: pd.DataFrame) -> str:
    window_note = _fingerprint_window_note(fingerprint_df)
    source_count = (
        fingerprint_df.groupby(["selected_sponge", "sensor", "comparison_group"], dropna=False)["value"].size().max()
    )
    evidence_note = (
        "Matched-tetO-referenced effect across intended sensor arms. Bars show source means; "
        "points show source-level estimates."
        if pd.notna(source_count) and int(source_count) > 1
        else "Matched-tetO-referenced effect across intended sensor arms. Points show the source-level estimate."
    )
    if window_note:
        return f"{window_note}; {evidence_note}"
    return evidence_note


def _fingerprint_window_note(fingerprint_df: pd.DataFrame) -> str:
    if not {"summary_window_start_h", "summary_window_end_h"}.issubset(fingerprint_df.columns):
        return ""
    window_pairs = (
        fingerprint_df[["summary_window_start_h", "summary_window_end_h"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["summary_window_start_h", "summary_window_end_h"], kind="stable")
    )
    if window_pairs.empty or len(window_pairs.index) != 1:
        return ""
    row = window_pairs.iloc[0]
    start_h = float(row["summary_window_start_h"])
    end_h = float(row["summary_window_end_h"])
    return f"Relevant-stress primary window: {start_h:.1f} to {end_h:.1f} h after stress addition"


def _wrap_hyphenated_plot_label(text: str, *, max_parts_per_line: int = 2) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    parts = [part for part in value.split("-") if part]
    if len(parts) <= max_parts_per_line:
        return value
    lines = ["-".join(parts[index : index + max_parts_per_line]) for index in range(0, len(parts), max_parts_per_line)]
    return "\n".join(lines)


def _metric_axis_label(metric: str) -> str:
    return retron_presentation.summary_metric_label(metric)


def _architecture_axis_label(architecture_x: str) -> str:
    if str(architecture_x) == "irrelevant_motif_count":
        return "Extra non-cognate motifs"
    return "Total motifs"


def _expected_axis_label(expected_mode: str, *, score_metric: str) -> str:
    if str(expected_mode) == "expected_best_single":
        return f"Best mono baseline ({_aggregate_score_axis_label(score_metric)})"
    return f"Sum of mono baselines ({_aggregate_score_axis_label(score_metric)})"


def _aggregate_score_axis_label(metric: str) -> str:
    labels = {
        "O_abs_AUC": "total effect",
        "S_abs_AUC": "scaled total effect",
        "O_AUC": "post-stress increment",
        "S_AUC": "scaled increment",
    }
    return labels.get(str(metric), _metric_axis_label(metric))


def _ordered_text(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values})
