from __future__ import annotations

import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reader.plotting.sinks import PlotFigure
from reader.plotting.style import PaletteBook, use_style

from .common import (
    annotate_points_smart,
    best_subplot_grid,
    bootstrap_linear_interval,
    bootstrap_mean_interval,
    colors_for,
    emit_plot_figure,
    require_columns,
    shared_numeric_limits,
    warn_if_empty,
)

_FAMILY_ORDER = {"mono": 0, "bi": 1, "tri": 2, "quad": 3, "control": 4}
_IPTG_ORDER = ("-IPTG", "+IPTG")
_TRACE_PRIMARY_WINDOW_METRICS = {"C", "D", "D_abs", "D_growth", "M", "O"}
_TRACE_INSET_METRICS = {"D", "D_abs"}


@dataclass(frozen=True)
class _TraceLegendPolicy:
    loc: str
    bbox_to_anchor: tuple[float, float]
    ncol_limit: int


@dataclass(frozen=True)
class _TraceSubplotPolicy:
    top: float
    bottom: float
    left: float
    right: float
    hspace: float
    wspace: float


@dataclass(frozen=True)
class _TraceFigurePolicy:
    default_figsize: tuple[float, float]
    sharex: bool
    sharey: bool | str
    xlabel_y: float
    title_y: float
    subtitle_y: float
    adjust_without_legend: _TraceSubplotPolicy
    adjust_with_legend: _TraceSubplotPolicy
    legend: _TraceLegendPolicy | None = None


@dataclass(frozen=True)
class _SummarySubplotPolicy:
    top: float
    bottom: float
    left: float
    right: float
    hspace: float
    wspace: float


@dataclass(frozen=True)
class _SummaryFigurePolicy:
    default_figsize: tuple[float, float]
    title_y: float
    subtitle_y: float
    adjust: _SummarySubplotPolicy
    xlabel: str | None = None
    xlabel_y: float | None = None
    xlabel_fontsize: float = 11.0
    ylabel: str | None = None
    ylabel_x: float | None = None
    ylabel_fontsize: float = 11.0


@dataclass(frozen=True)
class _HeatmapPanelSpec:
    metric: str
    title: str
    formula: str
    scale_group: str


@dataclass(frozen=True)
class _HeatmapPanelPayload:
    spec: _HeatmapPanelSpec
    pivot: pd.DataFrame


@dataclass(frozen=True)
class _DecisionCardRowPayload:
    row_idx: int
    sensor: str
    sponge: str
    stress: str
    relevant_sample: pd.DataFrame
    relevant_control: pd.DataFrame
    h2o_sample: pd.DataFrame
    h2o_control: pd.DataFrame
    panel_limits: tuple[float, float]


_LIBRARY_HEATMAP_PANELS: tuple[_HeatmapPanelSpec, ...] = (
    _HeatmapPanelSpec(
        metric="S_abs_AUC",
        title="Absolute on-target score\n(S_abs_AUC)",
        formula="S_abs_AUC = O_abs_AUC / |G_sensor|",
        scale_group="scaled",
    ),
    _HeatmapPanelSpec(
        metric="S_AUC",
        title="Post-stress incremental\nscore (S_AUC)",
        formula="S_AUC = O_AUC / |G_sensor|",
        scale_group="scaled",
    ),
    _HeatmapPanelSpec(
        metric="P_pre",
        title="Preload shift\n(P_pre)",
        formula="P_pre = delta_IPTG[R_pre - R_pre,tetO]",
        scale_group="delta",
    ),
)


def plot_retron_sponge_trace(
    *,
    trace: pd.DataFrame,
    output_dir: Path | None,
    metrics: Sequence[str],
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str = "tetO",
    include_control: bool = False,
    only_control: bool = False,
    relevant_only: bool = False,
    stress_order: Sequence[str] | None = None,
    panel_by: str = "stress",
    metric_label_map: Mapping[str, str] | None = None,
    fig_kwargs: dict | None = None,
) -> list[PlotFigure]:
    require_columns(
        trace,
        ["sensor", "sponge", "stress_condition", "time_from_stress", "metric", "value"],
        where="retron_sponge_trace",
    )
    fig_kwargs = fig_kwargs or {}
    selected_metrics = [str(metric) for metric in metrics]
    panel_mode = _validated_trace_panel_mode(panel_by)
    full_df = _filtered_retron_trace_frame(
        trace,
        control_name=control_name,
        include_control=include_control,
        only_control=only_control,
        relevant_only=relevant_only,
    )
    df = full_df[full_df["metric"].isin(selected_metrics)].copy()
    if warn_if_empty(df, where="retron_sponge_trace", detail="after metric/control filters"):
        return []

    sensors = _ordered(df["sensor"].tolist())
    figures: list[PlotFigure] = []
    for sensor in sensors:
        sensor_df = df[df["sensor"].astype(str) == sensor].copy()
        sensor_full_trace = full_df[full_df["sensor"].astype(str) == sensor].copy()
        stresses = _preferred_stresses(sensor_df["stress_condition"], stress_order=stress_order)
        display_post_stress_hours = float(fig_kwargs.get("display_post_stress_hours", 4.0))
        facet_by_sponge = panel_mode == "sponge" and not only_control and len(selected_metrics) == 1
        if facet_by_sponge:
            figures.extend(
                _plot_trace_sensor_faceted_by_sponge(
                    sensor=sensor,
                    sensor_df=sensor_df,
                    sensor_full_trace=sensor_full_trace,
                    stresses=stresses,
                    selected_metrics=selected_metrics,
                    title=title,
                    filename=filename,
                    output_dir=output_dir,
                    palette_book=palette_book,
                    metric_label_map=metric_label_map,
                    fig_kwargs=fig_kwargs,
                    control_name=control_name,
                    display_post_stress_hours=display_post_stress_hours,
                )
            )
            continue
        figures.extend(
            _plot_trace_sensor_grid(
                sensor=sensor,
                sensor_df=sensor_df,
                sensor_full_trace=sensor_full_trace,
                stresses=stresses,
                selected_metrics=selected_metrics,
                title=title,
                filename=filename,
                output_dir=output_dir,
                palette_book=palette_book,
                metric_label_map=metric_label_map,
                fig_kwargs=fig_kwargs,
                control_name=control_name,
                only_control=only_control,
                display_post_stress_hours=display_post_stress_hours,
            )
        )
    return figures


def _validated_trace_panel_mode(panel_by: str) -> str:
    panel_mode = str(panel_by or "stress").strip().lower()
    if panel_mode not in {"stress", "sponge"}:
        raise ValueError("retron_sponge_trace: panel_by supports only 'stress' or 'sponge'")
    return panel_mode


def _filtered_retron_trace_frame(
    trace: pd.DataFrame,
    *,
    control_name: str,
    include_control: bool,
    only_control: bool,
    relevant_only: bool,
) -> pd.DataFrame:
    full_df = trace.copy()
    if only_control:
        full_df = full_df[full_df["sponge"].astype(str) == str(control_name)]
    elif not include_control:
        full_df = full_df[full_df["sponge"].astype(str) != str(control_name)]
    if relevant_only:
        if "relevant_sensor_pair" not in full_df.columns:
            raise ValueError("retron_sponge_trace: relevant_only requires relevant_sensor_pair in the trace table.")
        full_df = full_df[full_df["relevant_sensor_pair"].fillna(False)]
    return full_df


def _trace_series_style(metric_df: pd.DataFrame) -> tuple[bool, list[str]]:
    has_iptg = "IPTG" in metric_df.columns and metric_df["IPTG"].notna().any()
    return has_iptg, _trace_iptg_levels(metric_df)


def _level_color_map(levels: Sequence[str], *, palette_book: PaletteBook | None) -> dict[str, str]:
    color_values = colors_for(max(1, len(levels)), palette_book)
    return {str(level): color_values[idx % len(color_values)] for idx, level in enumerate(levels)}


def _plot_trace_panel_groups(
    *,
    ax,
    frame: pd.DataFrame,
    full_trace: pd.DataFrame,
    metric: str,
    grouped_levels: Sequence[str],
    group_column: str,
    color_map: Mapping[str, str],
    has_iptg: bool,
    iptg_levels: Sequence[str],
    legend_handles: dict[str, object],
    fixed_stress_condition: str | None = None,
    fixed_sponge: str | None = None,
    iptg_labeler: Callable[[str, str], str],
) -> None:
    for level in grouped_levels:
        group_df = frame[frame[group_column].astype(str) == str(level)].copy()
        if group_df.empty:
            continue
        stress_condition = fixed_stress_condition if fixed_stress_condition is not None else str(level)
        sponge = fixed_sponge if fixed_sponge is not None else str(level)
        if has_iptg:
            for iptg in iptg_levels:
                subgroup = group_df[group_df["IPTG"].astype(str) == str(iptg)].copy()
                if subgroup.empty:
                    continue
                _plot_trace_line(
                    ax=ax,
                    df=subgroup,
                    full_trace=full_trace,
                    metric=metric,
                    stress_condition=stress_condition,
                    sponge=sponge,
                    color=color_map[str(level)],
                    label=iptg_labeler(str(level), str(iptg)),
                    linestyle="--" if str(iptg) == "+IPTG" else "-",
                    legend_handles=legend_handles,
                )
            continue
        _plot_trace_line(
            ax=ax,
            df=group_df,
            full_trace=full_trace,
            metric=metric,
            stress_condition=stress_condition,
            sponge=sponge,
            color=color_map[str(level)],
            label=str(level),
            linestyle="-",
            legend_handles=legend_handles,
        )


def _decorate_trace_axis(
    ax,
    *,
    metric: str,
    only_control: bool,
    trace: pd.DataFrame,
    stress_condition: str | None,
) -> None:
    if only_control:
        ax.axhline(0.0, color="#c7c7c7", linewidth=0.9, linestyle="-", alpha=0.95, zorder=0.7)
    elif _trace_zero_center(metric, only_control=False) is not None:
        ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
    if metric in _TRACE_PRIMARY_WINDOW_METRICS:
        _annotate_primary_window(ax, trace, stress_condition=stress_condition)
    ax.axvline(0.0, color="#9e9e9e", linewidth=0.9, linestyle="--", alpha=0.9, zorder=0.8)


def _trace_axis_bounds(
    *,
    metric_df: pd.DataFrame,
    metric: str,
    only_control: bool,
    full_trace: pd.DataFrame,
    display_post_stress_hours: float,
) -> tuple[tuple[float, float], tuple[float, float] | None] | None:
    if not metric_df["value"].notna().any():
        return None
    return (
        shared_numeric_limits(
            metric_df["value"].to_numpy(dtype=float, copy=False),
            center=_trace_zero_center(metric, only_control=only_control),
            pad_fraction=0.10,
            min_span=0.05,
        ),
        _trace_display_bounds(full_trace, max_post_stress_hours=display_post_stress_hours),
    )


def _apply_trace_axis_bounds(
    ax,
    *,
    y_limits: tuple[float, float],
    display_bounds: tuple[float, float] | None,
) -> None:
    ax.set_ylim(y_limits)
    if display_bounds is not None:
        ax.set_xlim(display_bounds)
    _annotate_stress_addition(ax)


def _set_trace_axis_box_aspect(ax) -> None:
    with suppress(Exception):
        ax.set_box_aspect(1.0)


def _add_trace_figure_legend(
    fig,
    legend_handles: Mapping[str, object],
    *,
    loc: str,
    bbox_to_anchor: tuple[float, float],
    ncol: int,
) -> None:
    if not legend_handles:
        return
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=False,
        title=None,
        borderaxespad=0.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def _faceted_trace_figure_policy(*, rows: int, cols: int) -> _TraceFigurePolicy:
    return _TraceFigurePolicy(
        default_figsize=(cols * 4.2, rows * 4.05),
        sharex=True,
        sharey=True,
        xlabel_y=0.08,
        title_y=0.988,
        subtitle_y=0.936,
        adjust_without_legend=_TraceSubplotPolicy(
            top=0.82,
            bottom=0.13,
            left=0.11,
            right=0.98,
            hspace=0.22,
            wspace=0.14,
        ),
        adjust_with_legend=_TraceSubplotPolicy(
            top=0.82,
            bottom=0.20,
            left=0.11,
            right=0.98,
            hspace=0.22,
            wspace=0.14,
        ),
        legend=_TraceLegendPolicy(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            ncol_limit=4,
        ),
    )


def _grid_trace_figure_policy(*, rows: int, cols: int, only_control: bool) -> _TraceFigurePolicy:
    if only_control:
        return _TraceFigurePolicy(
            default_figsize=(cols * 3.35, rows * 3.05),
            sharex=True,
            sharey="row",
            xlabel_y=0.09,
            title_y=0.988,
            subtitle_y=0.936,
            adjust_without_legend=_TraceSubplotPolicy(
                top=0.84,
                bottom=0.22,
                left=0.10,
                right=0.98,
                hspace=0.16,
                wspace=0.02,
            ),
            adjust_with_legend=_TraceSubplotPolicy(
                top=0.84,
                bottom=0.22,
                left=0.10,
                right=0.98,
                hspace=0.16,
                wspace=0.02,
            ),
            legend=_TraceLegendPolicy(
                loc="lower center",
                bbox_to_anchor=(0.5, 0.012),
                ncol_limit=2,
            ),
        )
    return _TraceFigurePolicy(
        default_figsize=(cols * 4.15, rows * 4.0),
        sharex=True,
        sharey="row",
        xlabel_y=0.02,
        title_y=0.988,
        subtitle_y=0.936,
        adjust_without_legend=_TraceSubplotPolicy(
            top=0.78,
            bottom=0.16,
            left=0.12,
            right=0.98,
            hspace=0.32,
            wspace=0.18,
        ),
        adjust_with_legend=_TraceSubplotPolicy(
            top=0.78,
            bottom=0.16,
            left=0.12,
            right=0.80,
            hspace=0.32,
            wspace=0.18,
        ),
        legend=_TraceLegendPolicy(
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            ncol_limit=1,
        ),
    )


def _trace_figure_size(
    *,
    fig_kwargs: Mapping[str, object],
    policy: _TraceFigurePolicy,
) -> tuple[float, float]:
    figsize = fig_kwargs.get("figsize", policy.default_figsize)
    return float(figsize[0]), float(figsize[1])


def _new_trace_figure(
    *,
    rows: int,
    cols: int,
    policy: _TraceFigurePolicy,
    fig_kwargs: Mapping[str, object],
):
    width, height = _trace_figure_size(fig_kwargs=fig_kwargs, policy=policy)
    return plt.subplots(
        rows,
        cols,
        figsize=(width, height),
        constrained_layout=False,
        squeeze=False,
        sharex=policy.sharex,
        sharey=policy.sharey,
    )


def _finalize_trace_figure(
    fig,
    *,
    legend_handles: Mapping[str, object],
    policy: _TraceFigurePolicy,
    fig_kwargs: Mapping[str, object],
    title: str,
    sensor: str,
    subtitle: str,
) -> None:
    _set_figure_header(
        fig,
        title=title,
        context=sensor,
        subtitle=subtitle,
        title_y=float(fig_kwargs.get("suptitle_y", policy.title_y)),
        subtitle_y=float(fig_kwargs.get("subtitle_y", policy.subtitle_y)),
    )
    fig.supxlabel("Time from stress addition (h)", y=policy.xlabel_y, fontsize=13)
    if legend_handles and policy.legend is not None:
        _add_trace_figure_legend(
            fig,
            legend_handles,
            loc=policy.legend.loc,
            bbox_to_anchor=policy.legend.bbox_to_anchor,
            ncol=min(policy.legend.ncol_limit, len(legend_handles)),
        )
    adjust = policy.adjust_with_legend if legend_handles and policy.legend is not None else policy.adjust_without_legend
    fig.subplots_adjust(
        top=adjust.top,
        bottom=adjust.bottom,
        left=adjust.left,
        right=adjust.right,
        hspace=adjust.hspace,
        wspace=adjust.wspace,
    )


def _interaction_summary_figure_policy(*, rows: int, cols: int) -> _SummaryFigurePolicy:
    return _SummaryFigurePolicy(
        default_figsize=(4.4 * cols, 3.9 * rows),
        title_y=0.988,
        subtitle_y=0.934,
        xlabel="IPTG and stress state",
        xlabel_y=0.02,
        xlabel_fontsize=11.0,
        adjust=_SummarySubplotPolicy(
            top=0.76,
            bottom=0.24,
            left=0.12,
            right=0.98,
            hspace=0.38,
            wspace=0.24,
        ),
    )


def _library_heatmap_figure_policy(*, max_rows: int, max_cols: int) -> _SummaryFigurePolicy:
    return _SummaryFigurePolicy(
        default_figsize=(
            max(13.0, 2.8 * len(_LIBRARY_HEATMAP_PANELS) + 0.58 * max_cols * len(_LIBRARY_HEATMAP_PANELS)),
            max(3.8, 2.2 + 0.34 * max_rows),
        ),
        title_y=0.988,
        subtitle_y=0.940,
        xlabel="Sponge",
        xlabel_y=0.03,
        xlabel_fontsize=13.0,
        ylabel="Sensor",
        ylabel_x=0.02,
        ylabel_fontsize=13.0,
        adjust=_SummarySubplotPolicy(
            top=0.78,
            bottom=0.18,
            left=0.10,
            right=0.99,
            hspace=0.12,
            wspace=0.03,
        ),
    )


def _decomposition_figure_policy(*, row_count: int) -> _SummaryFigurePolicy:
    return _SummaryFigurePolicy(
        default_figsize=(12.8, max(3.9, 3.2 * row_count)),
        title_y=0.988,
        subtitle_y=0.936,
        xlabel="Time from stress addition (h)",
        xlabel_y=0.06,
        xlabel_fontsize=12.0,
        adjust=_SummarySubplotPolicy(
            top=0.84,
            bottom=0.16,
            left=0.09,
            right=0.98,
            hspace=0.42,
            wspace=0.22,
        ),
    )


def _stress_modulation_figure_policy(*, row_count: int) -> _SummaryFigurePolicy:
    return _SummaryFigurePolicy(
        default_figsize=(8.9, max(4.8, 1.8 + 0.42 * row_count)),
        title_y=0.988,
        subtitle_y=0.938,
        adjust=_SummarySubplotPolicy(
            top=0.82,
            bottom=0.12,
            left=0.28,
            right=0.80,
            hspace=0.0,
            wspace=0.0,
        ),
    )


def _pareto_figure_policy() -> _SummaryFigurePolicy:
    return _SummaryFigurePolicy(
        default_figsize=(8.5, 5.5),
        title_y=0.98,
        subtitle_y=0.942,
        adjust=_SummarySubplotPolicy(
            top=0.88,
            bottom=0.11,
            left=0.12,
            right=0.80,
            hspace=0.0,
            wspace=0.0,
        ),
    )


def _summary_figure_size(
    *,
    fig_kwargs: Mapping[str, object],
    policy: _SummaryFigurePolicy,
) -> tuple[float, float]:
    figsize = fig_kwargs.get("figsize", policy.default_figsize)
    return float(fig_kwargs.get("figsize", policy.default_figsize)[0]), float(figsize[1])


def _new_summary_grid_figure(
    *,
    rows: int,
    cols: int,
    policy: _SummaryFigurePolicy,
    fig_kwargs: Mapping[str, object],
    sharex: bool | str = False,
    sharey: bool | str = False,
    gridspec_kw: Mapping[str, object] | None = None,
):
    width, height = _summary_figure_size(fig_kwargs=fig_kwargs, policy=policy)
    return plt.subplots(
        rows,
        cols,
        figsize=(width, height),
        constrained_layout=False,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=dict(gridspec_kw or {}),
    )


def _finalize_summary_figure(
    fig,
    *,
    policy: _SummaryFigurePolicy,
    fig_kwargs: Mapping[str, object],
    title: str,
    subtitle: str,
    context: str | None = None,
) -> None:
    _set_figure_header(
        fig,
        title=title,
        context=context,
        subtitle=subtitle,
        title_y=float(fig_kwargs.get("suptitle_y", policy.title_y)),
        subtitle_y=float(fig_kwargs.get("subtitle_y", policy.subtitle_y)),
    )
    if policy.xlabel and policy.xlabel_y is not None:
        fig.supxlabel(policy.xlabel, y=policy.xlabel_y, fontsize=policy.xlabel_fontsize)
    if policy.ylabel and policy.ylabel_x is not None:
        fig.supylabel(policy.ylabel, x=policy.ylabel_x, fontsize=policy.ylabel_fontsize)
    fig.subplots_adjust(
        top=policy.adjust.top,
        bottom=policy.adjust.bottom,
        left=policy.adjust.left,
        right=policy.adjust.right,
        hspace=policy.adjust.hspace,
        wspace=policy.adjust.wspace,
    )


def _plot_trace_sensor_faceted_by_sponge(
    *,
    sensor: str,
    sensor_df: pd.DataFrame,
    sensor_full_trace: pd.DataFrame,
    stresses: Sequence[str],
    selected_metrics: Sequence[str],
    title: str,
    filename: str | None,
    output_dir: Path | None,
    palette_book: PaletteBook | None,
    metric_label_map: Mapping[str, str] | None,
    fig_kwargs: Mapping[str, object],
    control_name: str,
    display_post_stress_hours: float,
) -> list[PlotFigure]:
    metric = str(selected_metrics[0])
    metric_df = sensor_df[sensor_df["metric"].astype(str) == metric].copy()
    sponge_levels = _sponge_levels(metric_df, control_name=control_name)
    rows, cols = best_subplot_grid(len(sponge_levels))
    policy = _faceted_trace_figure_policy(rows=rows, cols=cols)
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = _new_trace_figure(rows=rows, cols=cols, policy=policy, fig_kwargs=fig_kwargs)
        axes_flat = axes.ravel()
        legend_handles: dict[str, object] = {}
        has_iptg, iptg_levels = _trace_series_style(metric_df)
        stress_color_map = _level_color_map(stresses, palette_book=palette_book)
        for axis, sponge in zip(axes_flat, sponge_levels, strict=False):
            sponge_df = metric_df[metric_df["sponge"].astype(str) == sponge].copy()
            if sponge_df.empty:
                axis.set_visible(False)
                continue
            _plot_trace_panel_groups(
                ax=axis,
                frame=sponge_df,
                full_trace=sensor_full_trace,
                metric=metric,
                grouped_levels=stresses,
                group_column="stress_condition",
                color_map=stress_color_map,
                has_iptg=has_iptg,
                iptg_levels=iptg_levels,
                legend_handles=legend_handles,
                fixed_sponge=str(sponge),
                iptg_labeler=lambda stress, iptg: f"{stress}, {iptg}",
            )
            _decorate_trace_axis(axis, metric=metric, only_control=False, trace=sponge_df, stress_condition=None)
            axis.set_title(
                _wrap_hyphenated_label(str(sponge), max_parts_per_line=2),
                pad=6,
                fontweight="normal",
                fontsize=10,
            )
            axis.tick_params(axis="x", labelsize=8)
            axis.tick_params(axis="y", labelsize=8)
            _set_trace_axis_box_aspect(axis)
        for axis in axes_flat[len(sponge_levels) :]:
            axis.set_visible(False)
        bounds = _trace_axis_bounds(
            metric_df=metric_df,
            metric=metric,
            only_control=False,
            full_trace=sensor_full_trace,
            display_post_stress_hours=display_post_stress_hours,
        )
        if bounds is not None:
            y_limits, display_bounds = bounds
            for idx, axis in enumerate(axes_flat):
                if not axis.get_visible():
                    continue
                _apply_trace_axis_bounds(axis, y_limits=y_limits, display_bounds=display_bounds)
                if idx % cols == 0:
                    axis.set_ylabel(_metric_axis_label(metric, metric_label_map=metric_label_map), fontsize=13)
                    _add_axis_formula_tag(axis, _trace_metric_formula(metric))
                else:
                    axis.set_ylabel("")
            if metric in _TRACE_INSET_METRICS:
                for axis, sponge in zip(axes_flat, sponge_levels, strict=False):
                    if not axis.get_visible():
                        continue
                    sponge_metric_trace = metric_df[metric_df["sponge"].astype(str) == sponge].copy()
                    _add_trace_summary_inset(
                        axis,
                        trace=sponge_metric_trace,
                        metric=metric,
                        stress_order=stresses,
                        stress_color_map=stress_color_map,
                    )
        _finalize_trace_figure(
            fig,
            legend_handles=legend_handles,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            sensor=sensor,
            subtitle=_trace_figure_subtitle(selected_metrics, trace=sensor_full_trace),
        )
        return emit_plot_figure(
            fig=fig,
            filename=f"{filename or _slug(title)}__sensor={_slug(sensor)}",
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _plot_trace_sensor_grid(
    *,
    sensor: str,
    sensor_df: pd.DataFrame,
    sensor_full_trace: pd.DataFrame,
    stresses: Sequence[str],
    selected_metrics: Sequence[str],
    title: str,
    filename: str | None,
    output_dir: Path | None,
    palette_book: PaletteBook | None,
    metric_label_map: Mapping[str, str] | None,
    fig_kwargs: Mapping[str, object],
    control_name: str,
    only_control: bool,
    display_post_stress_hours: float,
) -> list[PlotFigure]:
    rows = len(selected_metrics)
    cols = max(1, len(stresses))
    policy = _grid_trace_figure_policy(rows=rows, cols=cols, only_control=only_control)
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = _new_trace_figure(rows=rows, cols=cols, policy=policy, fig_kwargs=fig_kwargs)
        legend_handles: dict[str, object] = {}
        for row_idx, metric in enumerate(selected_metrics):
            metric_df = sensor_df[sensor_df["metric"].astype(str) == metric].copy()
            sponge_levels = _sponge_levels(metric_df, control_name=control_name)
            color_map = _level_color_map(sponge_levels, palette_book=palette_book)
            has_iptg, iptg_levels = _trace_series_style(metric_df)
            for col_idx, stress in enumerate(stresses):
                ax = axes[row_idx][col_idx]
                panel = metric_df[metric_df["stress_condition"].astype(str) == stress].copy()
                if panel.empty:
                    ax.set_visible(False)
                    continue
                _plot_trace_panel_groups(
                    ax=ax,
                    frame=panel,
                    full_trace=sensor_full_trace,
                    metric=metric,
                    grouped_levels=sponge_levels,
                    group_column="sponge",
                    color_map=color_map,
                    has_iptg=has_iptg,
                    iptg_levels=iptg_levels,
                    legend_handles=legend_handles,
                    fixed_stress_condition=str(stress),
                    iptg_labeler=lambda sponge, iptg: f"{sponge} {iptg}",
                )
                _decorate_trace_axis(
                    ax,
                    metric=metric,
                    only_control=only_control,
                    trace=sensor_full_trace,
                    stress_condition=str(stress),
                )
                if row_idx == 0:
                    _set_axis_title(ax, _stress_panel_label(str(stress)), pad=6)
                else:
                    _set_axis_title(ax, "", pad=6)
                if col_idx == 0:
                    ax.set_ylabel(_metric_axis_label(metric, metric_label_map=metric_label_map), fontsize=13)
                    _add_axis_formula_tag(ax, _trace_metric_formula(metric))
                _set_trace_axis_box_aspect(ax)
            bounds = _trace_axis_bounds(
                metric_df=metric_df,
                metric=metric,
                only_control=only_control,
                full_trace=sensor_full_trace,
                display_post_stress_hours=display_post_stress_hours,
            )
            if bounds is not None:
                y_limits, display_bounds = bounds
                for ax in axes[row_idx]:
                    if ax.get_visible():
                        _apply_trace_axis_bounds(ax, y_limits=y_limits, display_bounds=display_bounds)
        _finalize_trace_figure(
            fig,
            legend_handles=legend_handles,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            sensor=sensor,
            subtitle=_trace_figure_subtitle(selected_metrics, trace=sensor_full_trace),
        )
        return emit_plot_figure(
            fig=fig,
            filename=f"{filename or _slug(title)}__sensor={_slug(sensor)}",
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _trace_iptg_levels(metric_df: pd.DataFrame) -> list[str]:
    if "IPTG" not in metric_df.columns or not metric_df["IPTG"].notna().any():
        return []
    observed = set(metric_df["IPTG"].dropna().astype(str))
    ordered_levels = [value for value in _IPTG_ORDER if value in observed]
    return ordered_levels or _ordered(metric_df["IPTG"].dropna().tolist())


def _trace_zero_center(metric: str, *, only_control: bool) -> float | None:
    if metric in {"B", "C", "D", "D_abs", "D_growth", "M", "O", "L_pre"}:
        return 0.0
    if only_control and metric in {"R", "mu"}:
        return 0.0
    return None


def plot_retron_sponge_summary(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None = None,
    output_dir: Path | None,
    view: str,
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str = "tetO",
    no_stress_label: str = "H2O",
    relevant_only: bool = True,
    metric: str | None = None,
    state_order: Sequence[str] | None = None,
    burden_metric: str = "D_growth_AUC",
    fig_kwargs: dict | None = None,
) -> list[PlotFigure]:
    require_columns(summary, ["sensor", "sponge", "metric", "value"], where="retron_sponge_summary")
    fig_kwargs = fig_kwargs or {}
    if view == "interaction":
        return _plot_retron_interaction_summary(
            summary=summary,
            trace=trace,
            output_dir=output_dir,
            title=title,
            filename=filename,
            palette_book=palette_book,
            control_name=control_name,
            no_stress_label=no_stress_label,
            relevant_only=relevant_only,
            metric=str(metric or "C_AUC"),
            state_order=state_order,
            fig_kwargs=fig_kwargs,
        )
    if view == "heatmap":
        return _plot_retron_library_heatmaps(
            summary=summary,
            trace=trace,
            output_dir=output_dir,
            title=title,
            filename=filename,
            control_name=control_name,
            no_stress_label=no_stress_label,
            relevant_only=relevant_only,
            fig_kwargs=fig_kwargs,
        )
    if view == "stress_modulation":
        return _plot_retron_stress_modulation(
            summary=summary,
            trace=trace,
            output_dir=output_dir,
            title=title,
            filename=filename,
            palette_book=palette_book,
            control_name=control_name,
            no_stress_label=no_stress_label,
            relevant_only=relevant_only,
            metric=str(metric or "M_AUC"),
            fig_kwargs=fig_kwargs,
        )
    if view == "decomposition":
        return _plot_retron_decomposition(
            summary=summary,
            trace=trace,
            output_dir=output_dir,
            title=title,
            filename=filename,
            control_name=control_name,
            relevant_only=relevant_only,
            fig_kwargs=fig_kwargs,
        )
    if view == "pareto":
        return _plot_retron_pareto(
            summary=summary,
            trace=trace,
            output_dir=output_dir,
            title=title,
            filename=filename,
            palette_book=palette_book,
            control_name=control_name,
            no_stress_label=no_stress_label,
            metric=str(metric or "S_abs_AUC"),
            burden_metric=burden_metric,
            fig_kwargs=fig_kwargs,
        )
    raise ValueError(f"retron_sponge_summary: unsupported view {view!r}")


def _plot_retron_interaction_summary(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir: Path | None,
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
    metric: str,
    state_order: Sequence[str] | None,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    if trace is None:
        raise ValueError("retron_interaction_summary: trace input is required to compute per-state uncertainty")
    require_columns(
        trace,
        ["stress_condition", "IPTG", "replicate_id", "time", "metric", "value"],
        where="retron_interaction_summary",
    )
    replicate_df = _interaction_replicate_summary(
        trace=trace,
        metric=metric,
        control_name=control_name,
        no_stress_label=no_stress_label,
        relevant_only=relevant_only,
    )
    if warn_if_empty(replicate_df, where="retron_interaction_summary", detail=metric):
        return []
    state_keys, state_label_map = _resolve_interaction_states(
        replicate_df=replicate_df,
        no_stress_label=no_stress_label,
        state_order=state_order,
    )
    figures: list[PlotFigure] = []
    state_palette = _interaction_state_palette(state_keys)
    for sensor in _ordered(replicate_df["sensor"].tolist()):
        sensor_df = replicate_df[replicate_df["sensor"].astype(str) == sensor].copy()
        sensor_trace = trace[trace["sensor"].astype(str) == sensor].copy()
        sponges = _sponge_levels(sensor_df, control_name=control_name)
        rows, cols = best_subplot_grid(len(sponges))
        policy = _interaction_summary_figure_policy(rows=rows, cols=cols)
        y_limits = shared_numeric_limits(
            sensor_df["value"].to_numpy(dtype=float, copy=False),
            center=0.0,
            pad_fraction=0.12,
            min_span=0.10,
        )
        with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
            fig, axes = _new_summary_grid_figure(
                rows=rows,
                cols=cols,
                policy=policy,
                fig_kwargs=fig_kwargs,
                sharey=True,
            )
            axes_flat = axes.ravel()
            for axis_index, (axis, sponge) in enumerate(zip(axes_flat, sponges, strict=False)):
                sponge_df = sensor_df[sensor_df["sponge"].astype(str) == sponge].copy()
                _plot_interaction_summary_axis(
                    axis,
                    sponge_df=sponge_df,
                    sponge=str(sponge),
                    state_keys=state_keys,
                    state_label_map=state_label_map,
                    state_palette=state_palette,
                    metric=metric,
                    y_limits=y_limits,
                    show_ylabel=axis_index % cols == 0,
                )
            for axis in axes_flat[len(sponges) :]:
                axis.set_visible(False)
            _finalize_summary_figure(
                fig,
                policy=policy,
                fig_kwargs=fig_kwargs,
                title=title,
                context=sensor,
                subtitle=_summary_metric_subtitle(metric, trace=sensor_trace),
            )
            figures.extend(
                emit_plot_figure(
                    fig=fig,
                    filename=f"{filename or _slug(title)}__sensor={_slug(sensor)}",
                    output_dir=output_dir,
                    fig_kwargs=fig_kwargs,
                )
            )
    return figures


def _interaction_state_palette(state_keys: Sequence[str]) -> dict[str, str]:
    base_palette = {
        "-IPTG/-stress": "#b0b0b0",
        "+IPTG/-stress": "#6f6f6f",
        "-IPTG/+stress": "#56B4E9",
        "+IPTG/+stress": "#0072B2",
    }
    return {str(state_key): base_palette.get(str(state_key), "#4c72b0") for state_key in state_keys}


def _interaction_state_values(sponge_df: pd.DataFrame, *, state_key: str) -> np.ndarray:
    state_df = sponge_df[sponge_df["state_key"] == str(state_key)].copy()
    values = pd.to_numeric(state_df["value"], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _interaction_state_interval(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean, lower, upper = bootstrap_mean_interval(
        values,
        ci=95.0,
        ci_boot=100,
        rng=np.random.default_rng(0),
    )
    return float(mean), float(lower), float(upper)


def _interaction_interval_frame(sponge_df: pd.DataFrame, *, state_keys: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for state_key in state_keys:
        mean, lower, upper = _interaction_state_interval(_interaction_state_values(sponge_df, state_key=str(state_key)))
        rows.append({"state_key": str(state_key), "mean": mean, "lower": lower, "upper": upper})
    return pd.DataFrame(rows)


def _plot_interaction_summary_axis(
    ax,
    *,
    sponge_df: pd.DataFrame,
    sponge: str,
    state_keys: Sequence[str],
    state_label_map: Mapping[str, str],
    state_palette: Mapping[str, str],
    metric: str,
    y_limits: tuple[float, float],
    show_ylabel: bool,
) -> None:
    x_positions = np.arange(len(state_keys), dtype=float)
    interval_frame = _interaction_interval_frame(sponge_df, state_keys=state_keys)
    means = pd.to_numeric(interval_frame["mean"], errors="coerce").to_numpy(dtype=float)
    lowers = pd.to_numeric(interval_frame["lower"], errors="coerce").to_numpy(dtype=float)
    uppers = pd.to_numeric(interval_frame["upper"], errors="coerce").to_numpy(dtype=float)
    ax.bar(
        x_positions,
        means,
        width=0.66,
        color=[state_palette.get(str(state_key), "#4c72b0") for state_key in state_keys],
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
    if np.isfinite(means).any():
        ax.errorbar(
            x_positions,
            means,
            yerr=np.vstack([means - lowers, uppers - means]),
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3.0,
            zorder=3,
        )
    for idx, state_key in enumerate(state_keys):
        state_values = _interaction_state_values(sponge_df, state_key=str(state_key))
        if state_values.size == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, num=state_values.size)
        ax.scatter(
            np.full(state_values.size, x_positions[idx], dtype=float) + jitter,
            state_values,
            s=22,
            alpha=0.7,
            color="#111111",
            zorder=4,
        )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            _format_interaction_state_label(state_label_map.get(str(state_key), str(state_key)))
            for state_key in state_keys
        ],
        rotation=0,
        ha="center",
    )
    ax.tick_params(axis="both", labelsize=8)
    ax.set_ylim(y_limits)
    ax.set_title(_wrap_hyphenated_label(sponge, max_parts_per_line=2), pad=6, fontweight="normal", fontsize=10)
    ax.set_ylabel(_summary_metric_label(metric) if show_ylabel else "", fontsize=11)
    with suppress(Exception):
        ax.set_box_aspect(1.0)


def build_retron_decomposition_frame(
    trace: pd.DataFrame,
    *,
    control_name: str,
    relevant_only: bool,
) -> pd.DataFrame:
    state_auc = _primary_window_auc_frame(trace, metric="R", control_name=control_name, relevant_only=relevant_only)
    if state_auc.empty:
        return pd.DataFrame(
            columns=[
                "sensor",
                "sponge",
                "stress_condition",
                "plate_id",
                "source_experiment_id",
                "source_label",
                "sample_minus_auc",
                "sample_plus_auc",
                "delta_real_auc",
                "control_minus_auc",
                "control_plus_auc",
                "delta_teto_auc",
                "delta_net_auc",
            ]
        )
    sample_rows = state_auc[~state_auc["is_control"]].copy()
    control_rows = state_auc[state_auc["is_control"]].copy()
    sample_group = _decomposition_group_columns(sample_rows, include_sponge=True)
    control_group = _decomposition_group_columns(control_rows, include_sponge=False)
    sample_pivot = _pivot_state_auc(sample_rows, index_columns=sample_group, value_prefix="sample")
    control_pivot = _pivot_state_auc(control_rows, index_columns=control_group, value_prefix="control")
    join_columns = [column for column in control_group if column in sample_pivot.columns]
    if join_columns:
        out = sample_pivot.merge(control_pivot, on=join_columns, how="left", validate="many_to_one")
    else:
        out = sample_pivot.assign(
            control_minus_auc=np.nan,
            control_plus_auc=np.nan,
        )
    out["delta_real_auc"] = out["sample_plus_auc"] - out["sample_minus_auc"]
    out["delta_teto_auc"] = out["control_plus_auc"] - out["control_minus_auc"]
    out["delta_net_auc"] = out["delta_real_auc"] - out["delta_teto_auc"]
    order = [column for column in ("sensor", "sponge", "stress_condition", "plate_id") if column in out.columns]
    if order:
        out = out.sort_values(order, kind="stable")
    return out.reset_index(drop=True)


def _matched_control_condition_frame(
    trace: pd.DataFrame,
    *,
    sensor: str,
    sponge: str,
    stress_condition: str,
    control_name: str,
    match_reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = trace[
        (trace["sensor"].astype(str) == str(sensor))
        & (trace["stress_condition"].astype(str) == str(stress_condition))
        & trace["IPTG"].notna()
    ].copy()
    sample = frame[frame["sponge"].astype(str) == str(sponge)].copy()
    control = frame[frame["sponge"].astype(str) == str(control_name)].copy()
    match_columns = [
        column
        for column in ("source_experiment_id", "source_label", "plate_id")
        if column in sample.columns and column in control.columns
    ]
    if match_columns:
        reference = match_reference if match_reference is not None else sample
        keys = reference[match_columns].drop_duplicates()
        if not keys.empty:
            sample = sample.merge(keys, on=match_columns, how="inner")
            control = control.merge(keys, on=match_columns, how="inner")
    return sample, control


def _decision_card_metric_frame(
    summary: pd.DataFrame,
    *,
    sensor: str,
    sponge: str,
    stress_condition: str,
) -> pd.DataFrame:
    required = {"sensor", "sponge", "metric", "value"}
    if not required.issubset(summary.columns):
        return pd.DataFrame(columns=["metric", "label", "color", "mean", "lower", "upper", "n"])
    metric_specs = (
        ("P_pre", "Preload shift", "#6f6f6f"),
        ("D_abs_AUC", "Total effect", "#0072B2"),
        ("D_AUC", "Post-stress increment", "#56B4E9"),
        ("D_growth_AUC", "Growth burden", "#D55E00"),
    )
    rows: list[dict[str, object]] = []
    for metric, label, color in metric_specs:
        metric_df = summary[
            (summary["sensor"].astype(str) == str(sensor))
            & (summary["sponge"].astype(str) == str(sponge))
            & (summary["metric"].astype(str) == metric)
        ].copy()
        if "stress_condition" in metric_df.columns:
            metric_df = metric_df[metric_df["stress_condition"].astype(str) == str(stress_condition)].copy()
        values = pd.to_numeric(metric_df["value"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            mean = lower = upper = np.nan
        elif values.size == 1:
            mean = lower = upper = float(values[0])
        else:
            mean, lower, upper = bootstrap_mean_interval(
                values,
                ci=95.0,
                ci_boot=100,
                rng=np.random.default_rng(0),
            )
        rows.append(
            {
                "metric": metric,
                "label": label,
                "color": color,
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def _window_read_count(trace: pd.DataFrame, *, flag_column: str) -> int | None:
    if trace.empty or flag_column not in trace.columns or "replicate_id" not in trace.columns:
        return None
    flagged = trace[trace[flag_column].fillna(False)].copy()
    if flagged.empty:
        return None
    counts = (
        flagged.groupby("replicate_id", dropna=False)["time_from_stress"].nunique().to_numpy(dtype=float, copy=False)
    )
    finite = counts[np.isfinite(counts)]
    if finite.size == 0:
        return None
    return int(round(float(np.median(finite))))


def _matched_control_relevant_trace_frame(
    trace: pd.DataFrame,
    *,
    metric: str,
    control_name: str,
    relevant_only: bool,
    where: str,
) -> pd.DataFrame:
    require_columns(
        trace,
        ["sensor", "sponge", "stress_condition", "IPTG", "metric", "value"],
        where=where,
    )
    frame = trace[trace["metric"].astype(str) == str(metric)].copy()
    frame = frame[frame["IPTG"].notna()]
    if frame.empty or not relevant_only:
        return frame
    _require_relevant_sensor_pair(frame, where=where)
    sample_mask = frame["sponge"].astype(str) != str(control_name)
    sample_frame = frame[sample_mask].copy()
    sample_frame = sample_frame[sample_frame["relevant_sensor_pair"].fillna(False)]
    if "is_relevant_stress" in sample_frame.columns:
        sample_frame = sample_frame[sample_frame["is_relevant_stress"].fillna(False)]
    control_frame = frame[~sample_mask].copy()
    match_columns = [
        column
        for column in ("source_experiment_id", "source_label", "plate_id", "sensor", "stress_condition")
        if column in sample_frame.columns and column in control_frame.columns
    ]
    if match_columns and not sample_frame.empty:
        control_frame = control_frame.merge(
            sample_frame[match_columns].drop_duplicates(),
            on=match_columns,
            how="inner",
        )
    elif "is_relevant_stress" in control_frame.columns:
        control_frame = control_frame[control_frame["is_relevant_stress"].fillna(False)]
    return pd.concat([sample_frame, control_frame], ignore_index=True)


def _plot_retron_decomposition(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir: Path | None,
    title: str,
    filename: str | None,
    control_name: str,
    relevant_only: bool,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    if trace is None:
        raise ValueError("retron_decomposition: trace input is required")
    trace_frame = _matched_control_relevant_trace_frame(
        trace,
        metric="R",
        control_name=control_name,
        relevant_only=relevant_only,
        where="retron_decomposition",
    )
    if warn_if_empty(trace_frame, where="retron_decomposition", detail="matched-control R traces"):
        return []
    r_trace = trace[(trace["metric"].astype(str) == "R") & trace["IPTG"].notna()].copy()
    support_frame = build_retron_decomposition_frame(trace, control_name=control_name, relevant_only=relevant_only)
    figures: list[PlotFigure] = []
    for sensor in _ordered(trace_frame["sensor"].astype(str).tolist()):
        sensor_df = trace_frame[trace_frame["sensor"].astype(str) == sensor].copy()
        sensor_trace = trace[trace["sensor"].astype(str) == sensor].copy()
        sensor_r_trace = r_trace[r_trace["sensor"].astype(str) == sensor].copy()
        row_payloads = _decision_card_row_payloads(
            sensor_df=sensor_df,
            sensor_r_trace=sensor_r_trace,
            control_name=control_name,
        )
        if not row_payloads:
            continue
        row_count = len(row_payloads)
        policy = _decomposition_figure_policy(row_count=row_count)
        display_bounds = _trace_display_bounds(
            sensor_trace,
            max_post_stress_hours=float(fig_kwargs.get("display_post_stress_hours", 4.0)),
        )
        with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
            fig, axes = _new_summary_grid_figure(
                rows=row_count,
                cols=3,
                policy=policy,
                fig_kwargs=fig_kwargs,
                gridspec_kw={"width_ratios": (1.65, 1.20, 1.0)},
            )
            if row_count == 1:
                axes = np.asarray([axes[0]])
            for row_payload in row_payloads:
                _render_decision_card_row(
                    axes=axes[row_payload.row_idx],
                    row=row_payload,
                    support_frame=support_frame,
                    summary=summary,
                    control_name=control_name,
                    display_bounds=display_bounds,
                )
            _finalize_summary_figure(
                fig,
                policy=policy,
                fig_kwargs=fig_kwargs,
                title=title,
                context=sensor,
                subtitle=_decomposition_subtitle(trace=sensor_trace),
            )
            figures.extend(
                emit_plot_figure(
                    fig=fig,
                    filename=f"{filename or _slug(title)}__sensor={_slug(sensor)}",
                    output_dir=output_dir,
                    fig_kwargs=fig_kwargs,
                )
            )
    return figures


def _decision_card_row_specs(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame(columns=["sponge", "stress_condition"])
    stress_levels = _preferred_stresses(
        sample_df["stress_condition"].astype(str).tolist(),
        stress_order=_ordered(sample_df["stress_condition"].astype(str).tolist()),
    )
    out = sample_df.assign(
        __stress_rank=pd.Categorical(
            sample_df["stress_condition"].astype(str),
            categories=stress_levels,
            ordered=True,
        )
    )
    return (
        out[["sponge", "stress_condition", "__stress_rank"]]
        .drop_duplicates()
        .sort_values(["__stress_rank", "sponge"], kind="stable")
        .reset_index(drop=True)
    )


def _decision_card_panel_limits(*frames: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat(
        [pd.to_numeric(frame["value"], errors="coerce") for frame in frames],
        ignore_index=True,
    ).to_numpy(dtype=float, copy=False)
    return shared_numeric_limits(values, center=None, pad_fraction=0.10, min_span=0.12)


def _decision_card_row_payloads(
    *,
    sensor_df: pd.DataFrame,
    sensor_r_trace: pd.DataFrame,
    control_name: str,
) -> list[_DecisionCardRowPayload]:
    sample_df = sensor_df[sensor_df["sponge"].astype(str) != str(control_name)].copy()
    row_specs = _decision_card_row_specs(sample_df)
    return [
        _decision_card_row_payload(
            row_idx=row_idx,
            sensor_r_trace=sensor_r_trace,
            sensor=str(sensor_df["sensor"].astype(str).iloc[0]),
            sponge=str(spec["sponge"]),
            stress=str(spec["stress_condition"]),
            control_name=control_name,
        )
        for row_idx, spec in row_specs.iterrows()
    ]


def _decision_card_row_payload(
    *,
    row_idx: int,
    sensor_r_trace: pd.DataFrame,
    sensor: str,
    sponge: str,
    stress: str,
    control_name: str,
) -> _DecisionCardRowPayload:
    relevant_sample, relevant_control = _matched_control_condition_frame(
        sensor_r_trace,
        sensor=sensor,
        sponge=sponge,
        stress_condition=stress,
        control_name=control_name,
    )
    h2o_sample, h2o_control = _matched_control_condition_frame(
        sensor_r_trace,
        sensor=sensor,
        sponge=sponge,
        stress_condition="H2O",
        control_name=control_name,
        match_reference=relevant_sample,
    )
    return _DecisionCardRowPayload(
        row_idx=row_idx,
        sensor=sensor,
        sponge=sponge,
        stress=stress,
        relevant_sample=relevant_sample,
        relevant_control=relevant_control,
        h2o_sample=h2o_sample,
        h2o_control=h2o_control,
        panel_limits=_decision_card_panel_limits(relevant_sample, relevant_control, h2o_sample, h2o_control),
    )


def _render_decision_card_row(
    *,
    axes: np.ndarray,
    row: _DecisionCardRowPayload,
    support_frame: pd.DataFrame,
    summary: pd.DataFrame,
    control_name: str,
    display_bounds: tuple[float, float] | None,
) -> None:
    _plot_decision_card_trace_axis(
        ax=axes[0],
        sample_panel=row.relevant_sample,
        control_panel=row.relevant_control,
        stress_condition=row.stress,
        row_idx=row.row_idx,
        sponge=row.sponge,
        control_name=control_name,
        panel_limits=row.panel_limits,
        display_bounds=display_bounds,
    )
    _plot_decision_card_trace_axis(
        ax=axes[1],
        sample_panel=row.h2o_sample,
        control_panel=row.h2o_control,
        stress_condition="H2O",
        row_idx=row.row_idx,
        sponge=row.sponge,
        control_name=control_name,
        panel_limits=row.panel_limits,
        display_bounds=display_bounds,
    )
    _plot_decision_card_summary_axis(
        ax=axes[2],
        row_idx=row.row_idx,
        sensor=row.sensor,
        sponge=row.sponge,
        stress=row.stress,
        relevant_sample=row.relevant_sample,
        summary=summary,
        support_frame=support_frame,
        control_name=control_name,
    )


def _plot_decision_card_trace_axis(
    *,
    ax: plt.Axes,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    stress_condition: str,
    row_idx: int,
    sponge: str,
    control_name: str,
    panel_limits: tuple[float, float],
    display_bounds: tuple[float, float] | None,
) -> None:
    legend_handles: dict[str, object] = {}
    line_specs = (
        ("Sample -IPTG", sample_panel[sample_panel["IPTG"].astype(str) == "-IPTG"].copy(), "#1f77b4", "-"),
        ("Sample +IPTG", sample_panel[sample_panel["IPTG"].astype(str) == "+IPTG"].copy(), "#1f77b4", "--"),
        (
            f"{control_name} -IPTG",
            control_panel[control_panel["IPTG"].astype(str) == "-IPTG"].copy(),
            "#8c8c8c",
            "-",
        ),
        (
            f"{control_name} +IPTG",
            control_panel[control_panel["IPTG"].astype(str) == "+IPTG"].copy(),
            "#8c8c8c",
            "--",
        ),
    )
    for label, line_df, color, linestyle in line_specs:
        if line_df.empty:
            continue
        _plot_trace_line(
            ax=ax,
            df=line_df,
            full_trace=line_df,
            metric="R",
            stress_condition=stress_condition,
            sponge=str(line_df["sponge"].astype(str).iloc[0]),
            color=color,
            label=label,
            linestyle=linestyle,
            legend_handles=legend_handles,
        )
    panel_trace = pd.concat([sample_panel, control_panel], ignore_index=True)
    _annotate_primary_window(ax, panel_trace, stress_condition=stress_condition)
    _annotate_stress_addition(ax)
    ax.grid(axis="both", color="#d9d9d9", linewidth=0.6, alpha=0.45)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(panel_limits)
    if display_bounds is not None:
        ax.set_xlim(display_bounds)
    if stress_condition == "H2O":
        ax.set_title("H2O context" if row_idx == 0 else "", pad=6, fontsize=10, fontweight="normal")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    else:
        title_lines = [
            _wrap_hyphenated_label(sponge, max_parts_per_line=2),
            _wrap_plot_text(stress_condition, width=18),
        ]
        ax.set_title("\n".join(title_lines), pad=6, fontsize=10, fontweight="normal")
        ax.set_ylabel(_metric_axis_label("R"), fontsize=11)
        if legend_handles:
            ax.legend(frameon=False, title=None, loc="lower left", fontsize=7.2)
    with suppress(Exception):
        ax.set_box_aspect(0.92 if stress_condition != "H2O" else 0.86)


def _plot_decision_card_summary_axis(
    *,
    ax: plt.Axes,
    row_idx: int,
    sensor: str,
    sponge: str,
    stress: str,
    relevant_sample: pd.DataFrame,
    summary: pd.DataFrame,
    support_frame: pd.DataFrame,
    control_name: str,
) -> None:
    metric_frame = _decision_card_metric_frame(summary, sensor=sensor, sponge=sponge, stress_condition=stress)
    positions = np.arange(len(metric_frame), dtype=float)
    x_values = pd.to_numeric(metric_frame["mean"], errors="coerce").to_numpy(dtype=float)
    lowers = pd.to_numeric(metric_frame["lower"], errors="coerce").to_numpy(dtype=float)
    uppers = pd.to_numeric(metric_frame["upper"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_values)
    summary_limits = shared_numeric_limits(
        np.concatenate(
            [
                x_values[np.isfinite(x_values)],
                lowers[np.isfinite(lowers)],
                uppers[np.isfinite(uppers)],
            ]
        )
        if valid.any()
        else np.array([0.0], dtype=float),
        center=0.0,
        pad_fraction=0.14,
        min_span=0.10,
    )
    for idx, row in metric_frame.iterrows():
        mean = pd.to_numeric(pd.Series([row["mean"]]), errors="coerce").iloc[0]
        lower = pd.to_numeric(pd.Series([row["lower"]]), errors="coerce").iloc[0]
        upper = pd.to_numeric(pd.Series([row["upper"]]), errors="coerce").iloc[0]
        if not np.isfinite(mean):
            continue
        ax.hlines(
            y=positions[idx],
            xmin=lower if np.isfinite(lower) else mean,
            xmax=upper if np.isfinite(upper) else mean,
            color=row["color"],
            linewidth=2.2,
            alpha=0.85,
            zorder=2,
        )
        ax.scatter(mean, positions[idx], s=42, color=row["color"], edgecolor="#222222", zorder=3)
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle=":")
    ax.set_xlim(summary_limits)
    ax.set_yticks(positions)
    ax.set_yticklabels(metric_frame["label"].tolist(), fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.55)
    ax.set_xlabel("Window summary", fontsize=9)
    if row_idx == 0:
        ax.set_title("Decision summary", pad=6, fontsize=10, fontweight="normal")
    support_text = _decision_card_support_text(
        sensor=sensor,
        sponge=sponge,
        stress=stress,
        relevant_sample=relevant_sample,
        summary=summary,
        support_frame=support_frame,
        control_name=control_name,
    )
    if support_text:
        ax.text(
            0.02,
            -0.22,
            support_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="#333333",
        )
    with suppress(Exception):
        ax.set_box_aspect(0.86)


def _decision_card_support_text(
    *,
    sensor: str,
    sponge: str,
    stress: str,
    relevant_sample: pd.DataFrame,
    summary: pd.DataFrame,
    support_frame: pd.DataFrame,
    control_name: str,
) -> str:
    matched_support = support_frame[
        (support_frame["sensor"].astype(str) == sensor)
        & (support_frame["sponge"].astype(str) == sponge)
        & (support_frame["stress_condition"].astype(str) == stress)
    ].copy()
    delta_real = pd.to_numeric(matched_support.get("delta_real_auc"), errors="coerce").mean()
    delta_teto = pd.to_numeric(matched_support.get("delta_teto_auc"), errors="coerce").mean()
    delta_net = pd.to_numeric(matched_support.get("delta_net_auc"), errors="coerce").mean()
    g_sensor = summary[
        (summary["sensor"].astype(str) == sensor)
        & (summary["sponge"].astype(str) == control_name)
        & (summary["metric"].astype(str) == "G_sensor")
    ]["value"]
    g_sensor_value = pd.to_numeric(g_sensor, errors="coerce").dropna().mean()
    paired_mask = (
        (summary["sensor"].astype(str) == sensor)
        & (summary["sponge"].astype(str) == sponge)
        & (summary["metric"].astype(str) == "D_abs_AUC")
    )
    if "stress_condition" in summary.columns:
        paired_mask = paired_mask & (summary["stress_condition"].astype(str) == stress)
    paired_count = int(summary.loc[paired_mask, "plate_id"].nunique()) if "plate_id" in summary.columns else 0
    pre_reads = _window_read_count(relevant_sample, flag_column="in_pre_window")
    post_reads = _window_read_count(relevant_sample, flag_column="in_primary_post_stress")
    support_lines: list[str] = []
    if np.isfinite(delta_real):
        support_lines.append(f"sample {delta_real:+.2f}")
    if np.isfinite(delta_teto):
        support_lines.append(f"tetO {delta_teto:+.2f}")
    if np.isfinite(delta_net):
        support_lines.append(f"net {delta_net:+.2f}")
    meta_parts = [f"n={paired_count}" if paired_count else None]
    if pre_reads is not None:
        meta_parts.append(f"pre={pre_reads}")
    if post_reads is not None:
        meta_parts.append(f"post={post_reads}")
    if np.isfinite(g_sensor_value):
        meta_parts.append(f"G={g_sensor_value:+.2f}")
    meta_parts = [part for part in meta_parts if part]
    if not support_lines and not meta_parts:
        return ""
    return "\n".join(
        [
            _wrap_plot_text("delta_sample / delta_tetO / D_abs_AUC", width=22),
            ", ".join(support_lines),
            " | ".join(meta_parts),
        ]
    ).strip()


def _plot_retron_library_heatmaps(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir: Path | None,
    title: str,
    filename: str | None,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    df = summary[summary["sponge"].astype(str) != str(control_name)].copy()
    if relevant_only:
        _require_relevant_sensor_pair(df, where="retron_library_heatmaps")
        df = df[df["relevant_sensor_pair"].fillna(False)]
    if warn_if_empty(df, where="retron_library_heatmaps", detail="after control/on-target filters"):
        return []
    panel_payloads = _library_heatmap_panel_payloads(df)
    panel_limits = _library_heatmap_limits(panel_payloads)
    max_rows = max(max(1, len(payload.pivot.index)) for payload in panel_payloads)
    max_cols = max(max(1, len(payload.pivot.columns)) for payload in panel_payloads)
    policy = _library_heatmap_figure_policy(max_rows=max_rows, max_cols=max_cols)
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = _new_summary_grid_figure(
            rows=1,
            cols=len(panel_payloads),
            policy=policy,
            fig_kwargs=fig_kwargs,
            sharey=True,
        )
        for panel_index, (ax, payload) in enumerate(zip(axes.ravel(), panel_payloads, strict=False)):
            _plot_library_heatmap_panel(
                ax,
                payload=payload,
                panel_limits=panel_limits,
                panel_index=panel_index,
            )
        _finalize_summary_figure(
            fig,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            subtitle=_library_heatmap_subtitle(trace=trace),
        )
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _library_heatmap_panel_frame(df: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    metric_df = df[df["metric"].astype(str) == str(metric)].copy()
    if metric_df.empty:
        return metric_df
    return metric_df[metric_df["is_relevant_stress"].fillna(False)].copy()


def _library_heatmap_panel_payloads(df: pd.DataFrame) -> list[_HeatmapPanelPayload]:
    return [
        _HeatmapPanelPayload(spec=spec, pivot=_pivot_summary(_library_heatmap_panel_frame(df, metric=spec.metric)))
        for spec in _LIBRARY_HEATMAP_PANELS
    ]


def _library_heatmap_limits(panel_payloads: Sequence[_HeatmapPanelPayload]) -> dict[str, tuple[float, float]]:
    limit_groups: dict[str, list[float]] = {"scaled": [], "delta": []}
    for payload in panel_payloads:
        values = pd.to_numeric(payload.pivot.to_numpy().ravel(), errors="coerce").tolist()
        limit_groups[payload.spec.scale_group].extend(values)
    return {
        scale_group: shared_numeric_limits(values, center=0.0, pad_fraction=0.02, min_span=0.10)
        for scale_group, values in limit_groups.items()
        if np.isfinite(pd.to_numeric(pd.Series(values), errors="coerce")).any()
    }


def _plot_library_heatmap_panel(
    ax,
    *,
    payload: _HeatmapPanelPayload,
    panel_limits: Mapping[str, tuple[float, float]],
    panel_index: int,
) -> None:
    if payload.pivot.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    sns.heatmap(
        payload.pivot,
        ax=ax,
        cmap="vlag",
        center=0.0,
        annot=True,
        fmt=".2f",
        cbar=False,
        square=False,
        linewidths=0.4,
        linecolor="#f0f0f0",
        annot_kws={"fontsize": 8.5},
        vmin=panel_limits.get(payload.spec.scale_group, (None, None))[0],
        vmax=panel_limits.get(payload.spec.scale_group, (None, None))[1],
    )
    _set_axis_title(ax, f"{payload.spec.title}\n{payload.spec.formula}", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    wrapped_columns = [_wrap_hyphenated_label(str(label), max_parts_per_line=2) for label in payload.pivot.columns]
    ax.set_xticklabels(wrapped_columns)
    ax.tick_params(axis="x", labelrotation=0, labelsize=9.0, pad=1)
    for label in ax.get_xticklabels():
        label.set_ha("center")
    if panel_index > 0:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.tick_params(axis="y", labelrotation=0, labelsize=10.0)


def _plot_retron_stress_modulation(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir: Path | None,
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
    metric: str,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    plot_df = _stress_modulation_plot_frame(
        summary=summary,
        metric=metric,
        control_name=control_name,
        relevant_only=relevant_only,
    )
    if warn_if_empty(plot_df, where="retron_stress_modulation", detail=metric):
        return []
    sensors = _ordered(plot_df["sensor"].tolist())
    sensor_colors = _level_color_map(sensors, palette_book=palette_book)
    policy = _stress_modulation_figure_policy(row_count=len(plot_df))
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = _new_summary_grid_figure(
            rows=1,
            cols=1,
            policy=policy,
            fig_kwargs=fig_kwargs,
        )
        ax = axes[0][0]
        chart_data = _stress_modulation_chart_data(plot_df)
        ax.set_xlim(
            shared_numeric_limits(
                chart_data["combined"] if chart_data["combined"].size else np.array([0.0], dtype=float),
                center=0.0,
                pad_fraction=0.10,
                min_span=0.10,
            )
        )
        _plot_stress_modulation_bars(ax, plot_df=plot_df, chart_data=chart_data, sensor_colors=sensor_colors)
        ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle=":")
        ax.set_xlabel(_summary_metric_label(metric), fontsize=11)
        ax.set_ylabel("")
        ax.set_yticks(chart_data["base_positions"])
        ax.set_yticklabels(chart_data["row_labels"])
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.55)
        _finalize_summary_figure(
            fig,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            subtitle=_summary_metric_subtitle(metric, trace=trace),
        )
        if chart_data["control_mask"].any() or chart_data["sample_mask"].any():
            ax.legend(
                frameon=False,
                title=None,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0.0,
            )
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _stress_modulation_chart_data(plot_df: pd.DataFrame) -> dict[str, object]:
    sample_values = pd.to_numeric(plot_df["sample_value"], errors="coerce").to_numpy(dtype=float)
    control_values = pd.to_numeric(plot_df["control_value"], errors="coerce").to_numpy(dtype=float)
    combined = np.concatenate([sample_values[np.isfinite(sample_values)], control_values[np.isfinite(control_values)]])
    return {
        "row_labels": [
            _stress_modulation_row_label(sensor=str(row.sensor), sponge=str(row.sponge))
            for row in plot_df.itertuples(index=False)
        ],
        "base_positions": np.arange(len(plot_df), dtype=float),
        "bar_height": 0.34,
        "sample_values": sample_values,
        "control_values": control_values,
        "sample_mask": np.isfinite(sample_values),
        "control_mask": np.isfinite(control_values),
        "sensor_labels": np.array(
            [str(sensor) for sensor in plot_df["sensor"].astype(str)],
            dtype=object,
        ),
        "combined": combined,
    }


def _plot_stress_modulation_bars(
    ax,
    *,
    plot_df: pd.DataFrame,
    chart_data: Mapping[str, object],
    sensor_colors: Mapping[str, str],
) -> None:
    base_positions = np.asarray(chart_data["base_positions"], dtype=float)
    bar_height = float(chart_data["bar_height"])
    sample_values = np.asarray(chart_data["sample_values"], dtype=float)
    control_values = np.asarray(chart_data["control_values"], dtype=float)
    sample_mask = np.asarray(chart_data["sample_mask"], dtype=bool)
    control_mask = np.asarray(chart_data["control_mask"], dtype=bool)
    sensor_labels = np.asarray(chart_data["sensor_labels"], dtype=object)
    edge_colors = np.array([sensor_colors.get(str(sensor), "#4c72b0") for sensor in sensor_labels], dtype=object)
    if control_mask.any():
        ax.barh(
            base_positions[control_mask] - bar_height / 2.0,
            control_values[control_mask],
            height=bar_height * 0.92,
            color="#f3ebe7",
            edgecolor=edge_colors[control_mask].tolist(),
            linewidth=0.9,
            hatch="//",
            label="tetO reference",
        )
    if sample_mask.any():
        ax.barh(
            base_positions[sample_mask] + bar_height / 2.0,
            sample_values[sample_mask],
            height=bar_height * 0.92,
            color=[sensor_colors.get(str(sensor), "#4c72b0") for sensor in plot_df.loc[sample_mask, "sensor"]],
            edgecolor="#222222",
            linewidth=0.5,
            label="Sample",
        )


def _plot_retron_pareto(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir: Path | None,
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str,
    no_stress_label: str,
    metric: str,
    burden_metric: str,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    relevant = _pareto_relevant_frame(summary, control_name=control_name)
    if warn_if_empty(relevant, where="retron_pareto", detail="after on-target filtering"):
        return []
    table = _pareto_summary_frame(
        summary=summary,
        relevant=relevant,
        control_name=control_name,
        metric=metric,
        burden_metric=burden_metric,
    )
    if warn_if_empty(table, where="retron_pareto", detail="after aggregation"):
        return []
    family_levels, color_map = _pareto_family_colors(table, palette_book=palette_book)
    sizes = _pareto_marker_sizes(table)
    policy = _pareto_figure_policy()
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = _new_summary_grid_figure(
            rows=1,
            cols=1,
            policy=policy,
            fig_kwargs=fig_kwargs,
        )
        ax = axes[0][0]
        _plot_pareto_points(ax, table=table, sizes=sizes, color_map=color_map)
        _format_pareto_axis(ax, metric=metric, burden_metric=burden_metric)
        legend_handles = _pareto_legend_handles(family_levels, color_map=color_map)
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                frameon=False,
                title=None,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0.0,
            )
        _finalize_summary_figure(
            fig,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            subtitle=_summary_metric_subtitle(metric, trace=trace),
        )
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _pareto_relevant_frame(summary: pd.DataFrame, *, control_name: str) -> pd.DataFrame:
    return summary[
        (summary["sponge"].astype(str) != str(control_name)) & summary["relevant_sensor_pair"].fillna(False)
    ].copy()


def _pareto_summary_frame(
    *,
    summary: pd.DataFrame,
    relevant: pd.DataFrame,
    control_name: str,
    metric: str,
    burden_metric: str,
) -> pd.DataFrame:
    score = relevant[relevant["metric"].astype(str) == str(metric)].groupby("sponge", dropna=False)["value"].mean()
    leak = relevant[relevant["metric"].astype(str) == "L_pre"].groupby("sponge", dropna=False)["value"].mean()
    family = relevant.groupby("sponge", dropna=False)["sponge_family_size"].agg(_first_non_null)
    burden_rows = summary[
        (summary["metric"].astype(str) == burden_metric)
        & (summary["sponge"].astype(str) != str(control_name))
        & summary["relevant_sensor_pair"].fillna(False)
    ][["sponge", "value"]].rename(columns={"value": "burden_value"})
    if burden_rows.empty:
        raise ValueError(f"retron_pareto: burden metric {burden_metric!r} is missing from the summary table")
    burden = burden_rows.groupby("sponge", dropna=False)["burden_value"].mean()
    return (
        pd.DataFrame({"on_target": score, "leakiness": leak, "burden": burden, "family": family})
        .reset_index()
        .dropna(subset=["on_target", "burden"])
    )


def _pareto_family_colors(
    table: pd.DataFrame,
    *,
    palette_book: PaletteBook | None,
) -> tuple[list[str], dict[str, str]]:
    family_levels = _ordered(
        table["family"].fillna("other").astype(str).tolist(),
        preferred=("mono", "bi", "tri", "quad", "other"),
    )
    return family_levels, _level_color_map(family_levels, palette_book=palette_book)


def _pareto_marker_sizes(table: pd.DataFrame) -> pd.Series:
    return 80.0 + 240.0 * table["leakiness"].abs().fillna(0.0)


def _plot_pareto_points(
    ax,
    *,
    table: pd.DataFrame,
    sizes: pd.Series,
    color_map: Mapping[str, str],
) -> None:
    ax.scatter(
        table["on_target"],
        table["burden"],
        s=sizes,
        c=[color_map.get(str(item), "#4c72b0") for item in table["family"].fillna("other")],
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    annotate_points_smart(
        ax=ax,
        points=[(float(row["on_target"]), float(row["burden"])) for _, row in table.iterrows()],
        labels=[_wrap_hyphenated_label(str(row["sponge"]), max_parts_per_line=2) for _, row in table.iterrows()],
    )


def _format_pareto_axis(ax, *, metric: str, burden_metric: str) -> None:
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle=":")
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
    ax.set_xlabel(f"Mean on-target effect across relevant sensors ({metric})")
    ax.set_ylabel(_burden_axis_label(burden_metric))
    ax.tick_params(axis="both", labelsize=7)
    with suppress(Exception):
        ax.set_box_aspect(1.0)


def _pareto_legend_handles(
    family_levels: Sequence[str],
    *,
    color_map: Mapping[str, str],
) -> list[plt.Line2D]:
    return [
        plt.Line2D([0], [0], marker="o", color="w", label=level, markerfacecolor=color_map[level], markersize=8)
        for level in family_levels
    ]


def _plot_trace_line(
    *,
    ax,
    df: pd.DataFrame,
    full_trace: pd.DataFrame,
    metric: str,
    stress_condition: str,
    sponge: str,
    color: str,
    label: str,
    linestyle: str,
    legend_handles: dict[str, object],
) -> None:
    summary = (
        _derived_trace_summary_frame(
            trace=full_trace,
            metric=metric,
            sponge=sponge,
            stress_condition=stress_condition,
        )
        if metric in {"D", "D_abs", "D_growth", "M", "O"}
        else _trace_summary_frame(df)
    )
    if summary.empty:
        return
    ax.fill_between(
        summary["time_from_stress"],
        summary["lower"],
        summary["upper"],
        alpha=0.16,
        color=color,
        linewidth=0.0,
        zorder=1,
    )
    (line,) = ax.plot(
        summary["time_from_stress"].to_numpy(dtype=float),
        summary["mean"].to_numpy(dtype=float),
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=label,
        zorder=2,
    )
    legend_handles.setdefault(label, line)


def _trace_display_bounds(
    trace: pd.DataFrame | None,
    *,
    max_post_stress_hours: float,
) -> tuple[float, float] | None:
    if trace is None or trace.empty or "time_from_stress" not in trace.columns:
        return None
    values = pd.to_numeric(trace["time_from_stress"], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lower = min(float(np.min(finite)), 0.0)
    upper_cap = max(0.0, float(max_post_stress_hours))
    positive = finite[finite >= 0.0]
    observed_upper = float(np.max(positive)) if positive.size else 0.0
    upper = max(observed_upper, upper_cap) if upper_cap > 0.0 else observed_upper
    if upper <= lower:
        upper = float(np.max(finite))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return lower, upper


def _trace_window_summary_frame(
    trace: pd.DataFrame,
    *,
    metric: str,
    stress_order: Sequence[str],
) -> pd.DataFrame:
    frame = trace[trace["metric"].astype(str) == str(metric)].copy() if "metric" in trace.columns else trace.copy()
    if "in_primary_post_stress" in frame.columns:
        frame = frame[frame["in_primary_post_stress"].fillna(False)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["stress_condition", "auc_value"])
    group_columns = [
        column for column in ("source_experiment_id", "plate_id", "stress_condition") if column in frame.columns
    ]
    rows: list[dict[str, object]] = []
    if not group_columns:
        group_columns = ["stress_condition"]
    for keys, group in frame.groupby(group_columns, dropna=False):
        record = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,), strict=False))
        ordered = group.sort_values("time_from_stress", kind="stable")
        times = pd.to_numeric(ordered["time_from_stress"], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(ordered["value"], errors="coerce").to_numpy(dtype=float)
        record["auc_value"] = _auc(times, values)
        rows.append(record)
    out = pd.DataFrame(rows)
    if out.empty or "stress_condition" not in out.columns:
        return pd.DataFrame(columns=["stress_condition", "auc_value"])
    out["stress_condition"] = out["stress_condition"].astype(str)
    summary = out.groupby("stress_condition", dropna=False)["auc_value"].mean().reset_index()
    if stress_order:
        summary["__stress_rank"] = pd.Categorical(
            summary["stress_condition"], categories=list(stress_order), ordered=True
        )
        summary = summary.sort_values(["__stress_rank", "stress_condition"], kind="stable").drop(
            columns="__stress_rank"
        )
    return summary.reset_index(drop=True)


def _add_trace_summary_inset(
    ax: plt.Axes,
    *,
    trace: pd.DataFrame,
    metric: str,
    stress_order: Sequence[str],
    stress_color_map: Mapping[str, str],
) -> None:
    summary_metric = {"D": "D_AUC", "D_abs": "D_abs_AUC"}.get(str(metric))
    if not summary_metric:
        return
    summary = _trace_window_summary_frame(trace, metric=metric, stress_order=stress_order)
    if summary.empty:
        return
    inset = ax.inset_axes([0.66, 0.08, 0.30, 0.30])
    positions = np.arange(len(summary), dtype=float)
    inset.bar(
        positions,
        pd.to_numeric(summary["auc_value"], errors="coerce").to_numpy(dtype=float),
        color=[stress_color_map.get(str(stress), "#4c72b0") for stress in summary["stress_condition"].astype(str)],
        edgecolor="#222222",
        linewidth=0.5,
        width=0.72,
        zorder=2,
    )
    inset.axhline(0.0, color="#777777", linewidth=0.8, linestyle=":")
    inset.set_xticks(positions)
    inset.set_xticklabels([_wrap_plot_text(str(label), width=10) for label in summary["stress_condition"]], rotation=0)
    inset.tick_params(axis="x", labelsize=6, pad=1)
    inset.tick_params(axis="y", labelsize=6, pad=1)
    inset.set_title(summary_metric, fontsize=7.2, pad=2, fontweight="normal")
    with suppress(Exception):
        inset.set_box_aspect(1.0)
    values = pd.to_numeric(summary["auc_value"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(values).any():
        limits = shared_numeric_limits(values, center=0.0, pad_fraction=0.14, min_span=0.08)
        inset.set_ylim(limits)
    for spine in inset.spines.values():
        spine.set_alpha(0.35)


def _derived_trace_summary_frame(
    *,
    trace: pd.DataFrame,
    metric: str,
    sponge: str,
    stress_condition: str,
) -> pd.DataFrame:
    source_metric = {"D": "C", "O": "C", "M": "C", "D_abs": "R", "D_growth": "mu"}.get(str(metric), "C")
    base_trace = trace[
        (trace["metric"].astype(str) == source_metric) & (trace["sponge"].astype(str) == str(sponge))
    ].copy()
    if base_trace.empty:
        return pd.DataFrame(columns=["time_from_stress", "mean", "lower", "upper"])
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(0)
    time_groups = base_trace.groupby("time_from_stress", dropna=False, sort=True)
    for time_value, time_group in time_groups:
        if metric in {"D", "O", "D_abs", "D_growth"}:
            stress_group = time_group[time_group["stress_condition"].astype(str) == str(stress_condition)].copy()
            plus = pd.to_numeric(
                stress_group[stress_group["IPTG"].astype(str) == "+IPTG"]["value"],
                errors="coerce",
            ).to_numpy(dtype=float)
            minus = pd.to_numeric(
                stress_group[stress_group["IPTG"].astype(str) == "-IPTG"]["value"],
                errors="coerce",
            ).to_numpy(dtype=float)
            mean, lower, upper = bootstrap_linear_interval(
                [plus, minus],
                coefficients=(1.0, -1.0),
                ci=95.0,
                ci_boot=100,
                rng=rng,
            )
            if metric == "O":
                expected_sign = float(
                    pd.to_numeric(stress_group["expected_decoy_sign"], errors="coerce").dropna().iloc[0]
                )
                mean, lower, upper = expected_sign * mean, expected_sign * lower, expected_sign * upper
        else:
            relevant = time_group[time_group["stress_condition"].astype(str) == str(stress_condition)].copy()
            baseline = time_group[~time_group["is_relevant_stress"].fillna(False)].copy()
            rel_plus = pd.to_numeric(
                relevant[relevant["IPTG"].astype(str) == "+IPTG"]["value"], errors="coerce"
            ).to_numpy(dtype=float)
            rel_minus = pd.to_numeric(
                relevant[relevant["IPTG"].astype(str) == "-IPTG"]["value"], errors="coerce"
            ).to_numpy(dtype=float)
            base_plus = pd.to_numeric(
                baseline[baseline["IPTG"].astype(str) == "+IPTG"]["value"], errors="coerce"
            ).to_numpy(dtype=float)
            base_minus = pd.to_numeric(
                baseline[baseline["IPTG"].astype(str) == "-IPTG"]["value"], errors="coerce"
            ).to_numpy(dtype=float)
            mean, lower, upper = bootstrap_linear_interval(
                [rel_plus, rel_minus, base_plus, base_minus],
                coefficients=(1.0, -1.0, -1.0, 1.0),
                ci=95.0,
                ci_boot=100,
                rng=rng,
            )
        if not np.isfinite(mean):
            continue
        rows.append(
            {
                "time_from_stress": float(time_value),
                "mean": float(mean),
                "lower": float(lower),
                "upper": float(upper),
            }
        )
    return pd.DataFrame(rows).sort_values("time_from_stress", kind="stable").reset_index(drop=True)


def _interaction_replicate_summary(
    *,
    trace: pd.DataFrame,
    metric: str,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
) -> pd.DataFrame:
    c_trace = trace[trace["metric"].astype(str) == "C"].copy()
    c_trace = c_trace[c_trace["sponge"].astype(str) != str(control_name)]
    if relevant_only:
        _require_relevant_sensor_pair(c_trace, where="retron_interaction_summary")
        c_trace = c_trace[c_trace["relevant_sensor_pair"].fillna(False)]
    rows: list[dict[str, object]] = []
    group_columns = [
        "plate_id",
        "sensor",
        "sponge",
        "genotype_id",
        "stress_condition",
        "IPTG",
        "replicate_id",
    ]
    for _, group in c_trace.groupby(group_columns, dropna=False):
        ordered = group.sort_values("time", kind="stable")
        values = pd.to_numeric(ordered["value"], errors="coerce").to_numpy(dtype=float)
        times = pd.to_numeric(ordered["time"], errors="coerce").to_numpy(dtype=float)
        if metric == "C_AUC":
            mask = ordered["in_primary_post_stress"].astype(bool).to_numpy()
            value = _auc(times[mask], values[mask])
        elif metric == "C_END":
            mask = ordered["in_endpoint_window"].astype(bool).to_numpy()
            value = np.nan if not mask.any() else float(np.nanmean(values[mask]))
        else:
            raise ValueError(f"retron_interaction_summary: unsupported metric {metric!r}")
        row = ordered.iloc[0]
        rows.append(
            {
                "plate_id": row["plate_id"],
                "sensor": row["sensor"],
                "sponge": row["sponge"],
                "genotype_id": row["genotype_id"],
                "stress_condition": row["stress_condition"],
                "IPTG": row["IPTG"],
                "replicate_id": row["replicate_id"],
                "state_key": _state_key(row, no_stress_label=no_stress_label),
                "state_label": _state_label(row, no_stress_label=no_stress_label),
                "value": value,
                "expected_decoy_sign": row.get("expected_decoy_sign"),
                "is_relevant_stress": row.get("is_relevant_stress"),
                "relevant_sensor_pair": row.get("relevant_sensor_pair"),
                "sponge_family_size": row.get("sponge_family_size"),
            }
        )
    return pd.DataFrame(rows)


def _primary_window_auc_frame(
    trace: pd.DataFrame,
    *,
    metric: str,
    control_name: str,
    relevant_only: bool,
) -> pd.DataFrame:
    require_columns(
        trace,
        [
            "sensor",
            "sponge",
            "stress_condition",
            "IPTG",
            "replicate_id",
            "time_from_stress",
            "metric",
            "value",
            "in_primary_post_stress",
        ],
        where="retron_primary_window_auc_frame",
    )
    frame = trace[trace["metric"].astype(str) == str(metric)].copy()
    frame = frame[frame["in_primary_post_stress"].fillna(False)]
    frame = frame[frame["IPTG"].notna()]
    if frame.empty:
        return pd.DataFrame()
    if relevant_only:
        frame = _matched_control_relevant_trace_frame(
            frame,
            metric=str(metric),
            control_name=control_name,
            relevant_only=True,
            where="retron_primary_window_auc_frame",
        )
        if frame.empty:
            return pd.DataFrame()
    group_columns = [
        column
        for column in (
            "source_experiment_id",
            "source_label",
            "plate_id",
            "sensor",
            "sponge",
            "stress_condition",
            "replicate_id",
            "IPTG",
            "configured_max_post_stress_hours",
        )
        if column in frame.columns
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        record = dict(zip(group_columns, keys, strict=False))
        ordered = group.sort_values("time_from_stress", kind="stable")
        times = pd.to_numeric(ordered["time_from_stress"], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(ordered["value"], errors="coerce").to_numpy(dtype=float)
        record["primary_window_auc"] = _auc(times, values)
        record["is_control"] = str(record.get("sponge", "")) == str(control_name)
        rows.append(record)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sensor"] = out["sensor"].astype(str)
    out["sponge"] = out["sponge"].astype(str)
    out["stress_condition"] = out["stress_condition"].astype(str)
    out["IPTG"] = out["IPTG"].astype(str)
    return out


def _decomposition_group_columns(frame: pd.DataFrame, *, include_sponge: bool) -> list[str]:
    columns = [
        column
        for column in ("source_experiment_id", "source_label", "plate_id", "sensor", "stress_condition")
        if column in frame.columns
    ]
    if include_sponge and "sponge" in frame.columns:
        columns.append("sponge")
    return columns


def _pivot_state_auc(frame: pd.DataFrame, *, index_columns: Sequence[str], value_prefix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*index_columns, f"{value_prefix}_minus_auc", f"{value_prefix}_plus_auc"])
    pivot = (
        frame.pivot_table(
            index=list(index_columns),
            columns="IPTG",
            values="primary_window_auc",
            aggfunc="mean",
        )
        .rename(columns={"-IPTG": f"{value_prefix}_minus_auc", "+IPTG": f"{value_prefix}_plus_auc"})
        .reset_index()
    )
    for expected in (f"{value_prefix}_minus_auc", f"{value_prefix}_plus_auc"):
        if expected not in pivot.columns:
            pivot[expected] = np.nan
    return pivot


def _trace_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["time_from_stress", "mean", "lower", "upper"])
    grouped = df.groupby("time_from_stress", dropna=False)["value"]
    rng = np.random.default_rng(0)
    rows: list[dict[str, float]] = []
    for time_value, series in grouped:
        mean, lower, upper = bootstrap_mean_interval(
            series.to_numpy(dtype=float, copy=False),
            ci=95.0,
            ci_boot=100,
            rng=rng,
        )
        rows.append(
            {
                "time_from_stress": float(time_value),
                "mean": mean,
                "lower": lower,
                "upper": upper,
            }
        )
    return pd.DataFrame(rows).sort_values("time_from_stress", kind="stable").reset_index(drop=True)


def _pivot_summary(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(index="sensor", columns="sponge", values="value", aggfunc="mean")
    if table.empty:
        return table
    row_order = _ordered(table.index.tolist())
    col_order = sorted(table.columns, key=_sponge_sort_key)
    return table.reindex(index=row_order, columns=col_order)


def _state_label(row: pd.Series, *, no_stress_label: str) -> str:
    iptg = str(row.get("IPTG") or "").strip() or "None"
    stress = str(row.get("stress_condition") or "").strip()
    return f"{stress or no_stress_label} / {iptg}"


def _state_key(row: pd.Series, *, no_stress_label: str) -> str:
    iptg = str(row.get("IPTG") or "").strip() or "-IPTG"
    stress = str(row.get("stress_condition") or "").strip()
    stress_key = "-stress" if not stress or stress == str(no_stress_label) else "+stress"
    return f"{iptg}/{stress_key}"


def _resolve_interaction_states(
    *,
    replicate_df: pd.DataFrame,
    no_stress_label: str,
    state_order: Sequence[str] | None,
) -> tuple[list[str], dict[str, str]]:
    del no_stress_label
    present_rows = (
        replicate_df[["state_key", "state_label"]]
        .drop_duplicates()
        .sort_values(["state_key", "state_label"], kind="stable")
    )
    state_label_map = {str(row["state_key"]): str(row["state_label"]) for _, row in present_rows.iterrows()}
    if state_order:
        ordered_keys = [str(item) for item in state_order if str(item) in state_label_map]
        ordered_keys.extend(key for key in state_label_map if key not in ordered_keys)
        return ordered_keys, state_label_map
    default_order = ("-IPTG/-stress", "+IPTG/-stress", "-IPTG/+stress", "+IPTG/+stress")
    ordered_keys = [key for key in default_order if key in state_label_map]
    ordered_keys.extend(key for key in state_label_map if key not in ordered_keys)
    return ordered_keys, state_label_map


def _format_interaction_state_label(label: str) -> str:
    stress, _, iptg = str(label).partition(" / ")
    if not iptg:
        return str(label)
    return f"{stress}\n{iptg}"


def _metric_axis_label(metric: str, *, metric_label_map: Mapping[str, str] | None = None) -> str:
    if metric_label_map and str(metric) in metric_label_map:
        return str(metric_label_map[str(metric)])
    labels = {
        "B": "Pre-stress-shifted ratio (B)",
        "C": "tetO-subtracted ratio (C)",
        "D": "IPTG-state effect (D)",
        "D_abs": "Absolute tetO-subtracted effect (D_abs)",
        "D_growth": "Growth burden (D_growth)",
        "M": "Stress-gated effect (M)",
        "O": "Signed effect (O)",
        "O_abs": "Signed absolute effect (O_abs)",
        "R": "log2(YFP/CFP) (R)",
        "mu": "d ln(OD600) / dt (mu)",
    }
    return labels.get(str(metric), f"Retron sponge metric ({metric})")


def _trace_metric_formula(metric: str) -> str:
    formulas = {
        "R": "R(t)=log2(YFP/CFP)",
        "B": "B(t)=R(t)-R_pre",
        "C": "C(t)=B(t)-B_tetO,matched(t)",
        "D": "D(t)=mean C(+IPTG)-mean C(-IPTG)",
        "D_abs": "D_abs(t)=delta_IPTG[R-R_tetO,matched]",
        "D_growth": "D_growth(t)=delta_IPTG[mu-mu_tetO,matched]",
        "M": "M(t)=D(sensor-matched stress)-D(H2O)",
        "O": "O(t)=expected_sign x D(t)",
        "O_abs": "O_abs(t)=expected_sign x D_abs(t)",
        "mu": "mu(t)=d ln(OD600) / dt",
    }
    return formulas.get(str(metric), str(metric))


def _trace_figure_subtitle(metrics: Sequence[str], *, trace: pd.DataFrame | None = None) -> str:
    del trace
    metric_ids = [str(metric) for metric in metrics]
    if len(metric_ids) == 1:
        return _trace_metric_formula(metric_ids[0])
    return ""


def _summary_metric_label(metric: str) -> str:
    labels = {
        "P_pre": "Preload shift (P_pre)",
        "C_AUC": "tetO-subtracted AUC (C_AUC)",
        "C_END": "tetO-subtracted endpoint (C_END)",
        "D_AUC": "IPTG-state effect AUC (D_AUC)",
        "D_abs_AUC": "Absolute tetO-subtracted AUC (D_abs_AUC)",
        "D_growth_AUC": "Growth burden AUC (D_growth_AUC)",
        "M_AUC": "Stress-gated AUC (M_AUC)",
        "O_abs_AUC": "Signed absolute AUC (O_abs_AUC)",
        "S_AUC": "Scaled effect (S_AUC)",
        "S_abs_AUC": "Scaled absolute effect (S_abs_AUC)",
    }
    return labels.get(str(metric), f"Retron sponge summary metric ({metric})")


def _summary_metric_formula(metric: str) -> str:
    formulas = {
        "P_pre": "P_pre = delta_IPTG[R_pre - R_pre,tetO]",
        "C_AUC": "C_AUC = AUC[C(t)] over the primary post-stress window",
        "C_END": "C_END = mean C(t) over the endpoint window",
        "D_AUC": "D_AUC = AUC[D(t)] over the primary post-stress window",
        "D_abs_AUC": "D_abs_AUC = AUC[D_abs(t)] over the primary post-stress window",
        "D_growth_AUC": "D_growth_AUC = AUC[D_growth(t)] over the primary post-stress window",
        "M_AUC": "M_AUC = AUC[M(t)] over the primary post-stress window",
        "O_abs_AUC": "O_abs_AUC = AUC[O_abs(t)] over the primary post-stress window",
        "S_AUC": "S_AUC = O_AUC / |G_sensor|",
        "S_abs_AUC": "S_abs_AUC = O_abs_AUC / |G_sensor|",
    }
    return formulas.get(str(metric), str(metric))


def _summary_metric_formula_compact(metric: str) -> str:
    formulas = {
        "P_pre": "P_pre = delta_IPTG[R_pre - R_pre,tetO]",
        "C_AUC": "C_AUC = AUC[C(t)]",
        "C_END": "C_END = END[C(t)]",
        "D_AUC": "D_AUC = AUC[D(t)]",
        "D_abs_AUC": "D_abs_AUC = AUC[D_abs(t)]",
        "D_growth_AUC": "D_growth_AUC = AUC[D_growth(t)]",
        "M_AUC": "M_AUC = AUC[M(t)]",
        "O_abs_AUC": "O_abs_AUC = AUC[O_abs(t)]",
        "S_AUC": "S_AUC = O_AUC / |G_sensor|",
        "S_abs_AUC": "S_abs_AUC = O_abs_AUC / |G_sensor|",
    }
    return formulas.get(str(metric), "")


def _summary_metric_subtitle(metric: str, *, trace: pd.DataFrame | None = None) -> str:
    notes: list[str] = []
    formula = _summary_metric_formula_compact(metric)
    if formula:
        notes.append(formula)
    summary_note = _primary_window_compact_note_from_trace(trace)
    if summary_note:
        notes.append(summary_note)
    if str(metric).endswith("_END"):
        endpoint_note = _endpoint_window_note_from_trace(trace)
        if endpoint_note:
            notes.append(endpoint_note)
    return "; ".join(notes)


def _annotate_primary_window(ax: plt.Axes, trace: pd.DataFrame, *, stress_condition: str | None) -> None:
    span = _primary_window_span_bounds(trace, stress_condition=stress_condition)
    if span is None:
        return
    start, end = span
    ax.axvspan(start, end, color="#f3b4b0", alpha=0.14, zorder=0.15, linewidth=0.0)


def _library_heatmap_subtitle(*, trace: pd.DataFrame | None = None) -> str:
    summary_note = _primary_window_compact_note_from_trace(trace)
    if not summary_note:
        return "Relevant-stress summaries over S_abs_AUC, S_AUC, and P_pre"
    return f"Relevant-stress summaries over S_abs_AUC, S_AUC, and P_pre; {summary_note}"


def _decomposition_subtitle(*, trace: pd.DataFrame | None = None) -> str:
    summary_note = _primary_window_compact_note_from_trace(trace)
    base = "R(t)=log2(YFP/CFP); compare preload, total effect, post-stress increment, and burden together"
    if not summary_note:
        return base
    return f"{base}; {summary_note}"


def _primary_window_compact_note_from_trace(trace: pd.DataFrame | None) -> str:
    if trace is None or trace.empty:
        return ""
    configured = _configured_primary_window_hours(trace)
    if configured is not None:
        return f"AUC and endpoint summarize the first {configured:.1f} h after stress addition"
    required = {"time_from_stress", "in_primary_post_stress"}
    if not required.issubset(trace.columns):
        return ""
    post = trace[trace["in_primary_post_stress"].fillna(False)].copy()
    if post.empty:
        return ""
    maxima = _window_group_maxima(post)
    finite = maxima[np.isfinite(maxima)]
    if finite.size == 0:
        return ""
    return f"AUC and endpoint summarize the first {float(finite.max()):.1f} h after stress addition"
    return ""


def _configured_primary_window_hours(trace: pd.DataFrame | None) -> float | None:
    if trace is None or trace.empty or "configured_max_post_stress_hours" not in trace.columns:
        return None
    values = pd.to_numeric(trace["configured_max_post_stress_hours"], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return float(values[0])


def _burden_axis_label(metric: str) -> str:
    if str(metric) == "D_growth_AUC":
        return "Mean growth burden (D_growth_AUC)"
    if str(metric) == "T_growth_AUC":
        return "Mean tetO growth burden (T_growth_AUC)"
    if str(metric) == "T_finalOD":
        return "Mean tetO endpoint burden (T_finalOD)"
    return f"Burden summary ({metric})"


def _endpoint_window_note_from_trace(trace: pd.DataFrame | None) -> str:
    if trace is None or trace.empty:
        return ""
    required = {"in_endpoint_window", "time"}
    if not required.issubset(trace.columns):
        return ""
    endpoint = trace[trace["in_endpoint_window"].fillna(False)].copy()
    if endpoint.empty:
        return ""
    count = _endpoint_time_count(endpoint)
    if count is None:
        return "Endpoint uses the last flagged reads inside that 4-hour range"
    noun = "read" if count == 1 else "reads"
    return f"Endpoint uses the last {count} flagged {noun} inside that 4-hour range"


def _window_group_maxima(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["time_from_stress"], errors="coerce")
    group_columns = [column for column in ("plate_id", "sensor", "stress_condition") if column in frame.columns]
    if not group_columns:
        finite = values[np.isfinite(values)]
        return finite.to_numpy(dtype=float, copy=False)
    grouped = frame.assign(__time_from_stress=values).groupby(group_columns, dropna=False)["__time_from_stress"].max()
    finite = pd.to_numeric(grouped, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return finite.to_numpy(dtype=float, copy=False)


def _primary_window_span_bounds(
    trace: pd.DataFrame | None,
    *,
    stress_condition: str | None,
) -> tuple[float, float] | None:
    if trace is None or trace.empty:
        return None
    required = {"stress_condition", "time_from_stress", "in_primary_post_stress"}
    if not required.issubset(trace.columns):
        return None
    post = trace[trace["in_primary_post_stress"].fillna(False)].copy()
    if stress_condition is not None:
        post = post[post["stress_condition"].astype(str) == str(stress_condition)].copy()
    if post.empty:
        return None
    configured = _configured_primary_window_hours(post)
    if configured is not None and configured > 0.0:
        end = float(configured)
    else:
        maxima = _window_group_maxima(post)
        finite = maxima[np.isfinite(maxima)]
        if finite.size == 0:
            return None
        end = float(finite.max())
    if end <= 0.0:
        return None
    return 0.0, end


def _window_group_minima(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["time_from_stress"], errors="coerce")
    group_columns = [column for column in ("plate_id", "sensor", "stress_condition") if column in frame.columns]
    if not group_columns:
        finite = values[np.isfinite(values)]
        return finite.to_numpy(dtype=float, copy=False)
    grouped = frame.assign(__time_from_stress=values).groupby(group_columns, dropna=False)["__time_from_stress"].min()
    finite = pd.to_numeric(grouped, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return finite.to_numpy(dtype=float, copy=False)


def _stress_modulation_plot_frame(
    *,
    summary: pd.DataFrame,
    metric: str,
    control_name: str,
    relevant_only: bool,
) -> pd.DataFrame:
    df = summary[summary["metric"].astype(str) == str(metric)].copy()
    sample_df = df[df["sponge"].astype(str) != str(control_name)].copy()
    if relevant_only:
        _require_relevant_sensor_pair(sample_df, where="retron_stress_modulation")
        sample_df = sample_df[sample_df["relevant_sensor_pair"].fillna(False)]
    if sample_df.empty:
        return pd.DataFrame(columns=["sensor", "sponge", "sample_value", "control_value"])
    control_df = df[df["sponge"].astype(str) == str(control_name)].copy()
    join_keys = [
        column
        for column in ("sensor", "stress_condition")
        if column in sample_df.columns
        and column in control_df.columns
        and sample_df[column].notna().any()
        and control_df[column].notna().any()
    ]
    if not join_keys:
        join_keys = ["sensor"]
    control_lookup = control_df.groupby(join_keys, dropna=False)["value"].mean().rename("control_value").reset_index()
    plot_df = sample_df.merge(control_lookup, on=join_keys, how="left")
    plot_df["sample_value"] = pd.to_numeric(plot_df["value"], errors="coerce")
    plot_df["control_value"] = pd.to_numeric(plot_df["control_value"], errors="coerce")
    order = [column for column in ("sensor", "sponge") if column in plot_df.columns]
    if order:
        plot_df = plot_df.sort_values(order, kind="stable")
    keep = [column for column in ("sensor", "sponge", "sample_value", "control_value") if column in plot_df.columns]
    return plot_df[keep].reset_index(drop=True)


def _stress_modulation_row_label(*, sensor: str, sponge: str) -> str:
    return f"{sensor}\n{_wrap_hyphenated_label(sponge, max_parts_per_line=2)}"


def _endpoint_time_count(frame: pd.DataFrame) -> int | None:
    group_columns = [column for column in ("plate_id", "sensor", "stress_condition") if column in frame.columns]
    if not group_columns:
        count = int(pd.to_numeric(frame["time"], errors="coerce").dropna().nunique())
        return count or None
    counts = (
        frame.assign(__time=pd.to_numeric(frame["time"], errors="coerce"))
        .groupby(group_columns, dropna=False)["__time"]
        .nunique()
    )
    counts = counts[counts > 0]
    if counts.empty:
        return None
    modes = counts.mode(dropna=True)
    if modes.empty:
        return int(counts.iloc[0])
    return int(modes.iloc[0])


def _format_hour_range(values: np.ndarray) -> str:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return ""
    low = float(np.min(finite))
    high = float(np.max(finite))
    if np.isclose(low, high, atol=0.05):
        return f"{high:.1f} h"
    return f"{low:.1f}-{high:.1f} h"


def _stress_panel_label(stress: str) -> str:
    if not stress:
        return "Stress not declared"
    return _wrap_plot_text(str(stress), width=20)


def _set_axis_title(ax, title: str, *, pad: float = 8.0) -> None:
    ax.set_title(_wrap_plot_text(str(title), width=24), pad=pad, fontweight="normal", fontsize=10)


def _add_axis_formula_tag(ax, formula: str, *, x: float = 0.02, y: float = 0.98) -> None:
    formula_text = str(formula or "").strip()
    if not formula_text:
        return
    ax.text(
        x,
        y,
        _wrap_plot_text(formula_text, width=28),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.22},
        zorder=3.2,
    )


def _annotate_stress_addition(ax) -> None:
    if any(text.get_text() == "Stress addition" for text in ax.texts):
        return
    x_limits = ax.get_xlim()
    if len(x_limits) != 2 or not np.isfinite(x_limits).all() or not (x_limits[0] <= 0.0 <= x_limits[1]):
        return
    ax.annotate(
        "Stress addition",
        xy=(0.0, 0.08),
        xycoords=ax.get_xaxis_transform(),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#666666",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.25},
        zorder=3.5,
    )


def _set_figure_header(
    fig,
    *,
    title: str,
    context: str | None = None,
    subtitle: str | None = None,
    title_y: float = 0.98,
    subtitle_y: float = 0.945,
) -> None:
    _set_figure_title(fig, title=title, context=context, y=title_y)
    subtitle_text = str(subtitle or "").strip()
    if subtitle_text:
        fig.text(
            0.5,
            subtitle_y,
            _wrap_plot_text(subtitle_text, width=96),
            ha="center",
            va="top",
            color="#333333",
            fontsize=8.5,
        )


def _set_figure_title(fig, *, title: str, context: str | None = None, y: float = 0.98) -> None:
    figure_title = str(title).strip()
    context_text = str(context or "").strip()
    if context_text:
        figure_title = f"{figure_title} · {context_text}"
    fig.suptitle(_wrap_plot_text(figure_title, width=86), y=y, x=0.5, ha="center", fontweight="normal", fontsize=14)


def _wrap_plot_text(text: str, *, width: int) -> str:
    value = str(text or "").strip()
    if not value or len(value) <= width:
        return value
    if "\n" in value:
        return "\n".join(_wrap_plot_text(line, width=width) for line in value.splitlines())
    parts = [part.strip() for part in value.split(";") if part.strip()]
    if len(parts) > 1:
        lines: list[str] = []
        current = ""
        for part in parts:
            segment = part if not current else f"{current}; {part}"
            if len(segment) <= width:
                current = segment
                continue
            if current:
                lines.append(current)
            current = part
        if current:
            lines.append(current)
        return "\n".join(lines)
    return textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)


def _wrap_hyphenated_label(label: str, *, max_parts_per_line: int = 2) -> str:
    parts = [part for part in str(label).split("-") if part]
    if len(parts) <= max_parts_per_line:
        return str(label)
    lines = ["-".join(parts[index : index + max_parts_per_line]) for index in range(0, len(parts), max_parts_per_line)]
    return "\n".join(lines)


def _first_non_null(series: pd.Series) -> str:
    for value in series:
        if pd.notna(value):
            return str(value)
    return "other"


def _auc(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) == 0 or len(values) == 0:
        return float("nan")
    finite = np.isfinite(times) & np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.trapezoid(values[finite], times[finite]))


def _require_relevant_sensor_pair(df: pd.DataFrame, *, where: str) -> None:
    if "relevant_sensor_pair" not in df.columns:
        raise ValueError(f"{where}: relevant_sensor_pair is required for on-target filtering")


def _preferred_stresses(values: Iterable[object], *, stress_order: Sequence[str] | None) -> list[str]:
    preferred = [str(value) for value in (stress_order or []) if str(value).strip()]
    return _ordered(values, preferred=preferred or ("H2O",))


def _sponge_levels(df: pd.DataFrame, *, control_name: str) -> list[str]:
    levels = df["sponge"].dropna().astype(str).unique().tolist()
    return sorted(levels, key=lambda item: _sponge_sort_key(item, control_name=control_name))


def _sponge_sort_key(value: str, *, control_name: str = "tetO") -> tuple[int, str]:
    if value == control_name:
        return (_FAMILY_ORDER["control"], value)
    parts = [part for part in str(value).split("-") if part]
    size = {1: "mono", 2: "bi", 3: "tri", 4: "quad"}.get(len(parts), "other")
    return (_FAMILY_ORDER.get(size, 99), str(value))


def _ordered(values: Iterable[object], preferred: Sequence[str] | None = None) -> list[str]:
    seen = {str(value) for value in values if pd.notna(value)}
    ordered = [item for item in (preferred or []) if item in seen]
    ordered.extend(sorted(item for item in seen if item not in ordered))
    return ordered


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value)).strip("_").lower()
