from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis._retron_sponge_contract import DEFAULT_PRIMARY_POST_STRESS_HOURS
from reader.plotting.sinks import PlotFigure
from reader.plotting.style import PaletteBook, use_style

from . import _retron_sponge_presentation as retron_presentation
from ._retron_sponge_trace_support import (
    annotate_primary_window as _annotate_primary_window,
)
from ._retron_sponge_trace_support import (
    annotate_stress_addition as _annotate_stress_addition,
)
from ._retron_sponge_trace_support import (
    trace_display_bounds as _trace_display_bounds,
)
from ._retron_sponge_trace_support import (
    trace_summary_frame as _trace_summary_frame,
)
from .common import (
    best_subplot_grid,
    bootstrap_linear_interval,
    emit_plot_figure,
    require_columns,
    shared_numeric_limits,
    warn_if_empty,
)
from .retron_sponge_summary_views import SUMMARY_VIEW_RENDERERS, _RetronSummaryPlotRequest
from .retron_sponge_summary_views.decomposition import build_retron_decomposition_frame
from .retron_sponge_summary_views.shared import (
    _auc,
    _level_color_map,
    _ordered,
    _preferred_stresses,
    _set_axis_title,
    _set_figure_header,
    _slug,
    _sponge_levels,
    _wrap_hyphenated_label,
    _wrap_plot_text,
)

_IPTG_ORDER = ("-IPTG", "+IPTG")

__all__ = [
    "build_retron_decomposition_frame",
    "plot_retron_sponge_summary",
    "plot_retron_sponge_trace",
]


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
        display_post_stress_hours = _resolved_display_post_stress_hours(
            trace=sensor_full_trace,
            fig_kwargs=fig_kwargs,
        )
        if only_control:
            figures.extend(
                _plot_control_trace_compact(
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
                    display_post_stress_hours=display_post_stress_hours,
                )
            )
            continue
        facet_by_sponge = panel_mode == "sponge" and not only_control and len(selected_metrics) == 1
        if facet_by_sponge and str(selected_metrics[0]) in {"D", "D_abs"}:
            continue
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
    if (
        panel_mode == "sponge"
        and not only_control
        and len(selected_metrics) == 1
        and str(selected_metrics[0]) in {"D", "D_abs"}
    ):
        figures = _plot_trace_all_sensors_faceted_by_pair(
            trace_df=df,
            full_trace=full_df,
            selected_metric=str(selected_metrics[0]),
            title=title,
            filename=filename,
            output_dir=output_dir,
            palette_book=palette_book,
            metric_label_map=metric_label_map,
            fig_kwargs=fig_kwargs,
            control_name=control_name,
            stress_order=stress_order,
            display_post_stress_hours=display_post_stress_hours,
        )
    return figures


def _validated_trace_panel_mode(panel_by: str) -> str:
    panel_mode = str(panel_by or "stress").strip().lower()
    if panel_mode not in {"stress", "sponge"}:
        raise ValueError("retron_sponge_trace: panel_by supports only 'stress' or 'sponge'")
    return panel_mode


def _resolved_display_post_stress_hours(
    *,
    trace: pd.DataFrame | None,
    fig_kwargs: Mapping[str, object],
) -> float:
    explicit = pd.to_numeric(pd.Series([fig_kwargs.get("display_post_stress_hours")]), errors="coerce").iloc[0]
    if np.isfinite(explicit) and explicit > 0.0:
        return float(explicit)
    span = retron_presentation.primary_window_span_bounds(trace, stress_condition=None)
    if span is not None and np.isfinite(span[1]) and span[1] > 0.0:
        return float(span[1])
    return float(DEFAULT_PRIMARY_POST_STRESS_HOURS)


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
    if retron_presentation.should_annotate_primary_window(metric):
        _annotate_primary_window(ax, trace, stress_condition=stress_condition)


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
        default_figsize=(cols * 3.95, rows * 3.95),
        sharex=True,
        sharey=True,
        xlabel_y=0.036,
        title_y=0.988,
        subtitle_y=0.948,
        adjust_without_legend=_TraceSubplotPolicy(
            top=0.86,
            bottom=0.10,
            left=0.09,
            right=0.99,
            hspace=0.18,
            wspace=0.02,
        ),
        adjust_with_legend=_TraceSubplotPolicy(
            top=0.86,
            bottom=0.27,
            left=0.09,
            right=0.99,
            hspace=0.18,
            wspace=0.02,
        ),
        legend=_TraceLegendPolicy(
            loc="lower left",
            bbox_to_anchor=(0.015, 0.006),
            ncol_limit=1,
        ),
    )


def _faceted_effect_trace_figure_policy(*, rows: int, cols: int) -> _TraceFigurePolicy:
    return _TraceFigurePolicy(
        default_figsize=(cols * 4.05, rows * 4.10),
        sharex=True,
        sharey=True,
        xlabel_y=0.036,
        title_y=0.988,
        subtitle_y=0.948,
        adjust_without_legend=_TraceSubplotPolicy(
            top=0.86,
            bottom=0.14,
            left=0.11,
            right=0.99,
            hspace=0.30,
            wspace=0.10,
        ),
        adjust_with_legend=_TraceSubplotPolicy(
            top=0.86,
            bottom=0.185,
            left=0.11,
            right=0.99,
            hspace=0.30,
            wspace=0.10,
        ),
        legend=_TraceLegendPolicy(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol_limit=6,
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
        xlabel_y=0.028,
        title_y=0.988,
        subtitle_y=0.942,
        adjust_without_legend=_TraceSubplotPolicy(
            top=0.80,
            bottom=0.14,
            left=0.10,
            right=0.985,
            hspace=0.26,
            wspace=0.10,
        ),
        adjust_with_legend=_TraceSubplotPolicy(
            top=0.80,
            bottom=0.14,
            left=0.10,
            right=0.80,
            hspace=0.26,
            wspace=0.10,
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
    include_shared_xlabel: bool = True,
) -> None:
    _set_figure_header(
        fig,
        title=title,
        context=sensor,
        subtitle=subtitle,
        title_y=float(fig_kwargs.get("suptitle_y", policy.title_y)),
        subtitle_y=float(fig_kwargs.get("subtitle_y", policy.subtitle_y)),
    )
    if include_shared_xlabel:
        fig.supxlabel("Time from stress addition (h)", y=policy.xlabel_y, fontsize=13.5)
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
                fontsize=10.6,
            )
            axis.tick_params(axis="x", labelsize=8.6)
            axis.tick_params(axis="y", labelsize=8.6)
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
                    axis.set_ylabel(
                        retron_presentation.metric_axis_label(metric, metric_label_map=metric_label_map),
                        fontsize=13.2,
                    )
                else:
                    axis.set_ylabel("")
            if retron_presentation.has_trace_summary_inset(metric):
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
        trace_subtitle = (
            retron_presentation.trace_text_spec(selected_metrics[0]).figure_subtitle()
            if len(selected_metrics) == 1
            else ""
        )
        _finalize_trace_figure(
            fig,
            legend_handles=legend_handles,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            sensor=sensor,
            subtitle=trace_subtitle,
        )
        return emit_plot_figure(
            fig=fig,
            filename=f"{filename or _slug(title)}__sensor={_slug(sensor)}",
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _plot_trace_all_sensors_faceted_by_pair(
    *,
    trace_df: pd.DataFrame,
    full_trace: pd.DataFrame,
    selected_metric: str,
    title: str,
    filename: str | None,
    output_dir: Path | None,
    palette_book: PaletteBook | None,
    metric_label_map: Mapping[str, str] | None,
    fig_kwargs: Mapping[str, object],
    control_name: str,
    stress_order: Sequence[str] | None,
    display_post_stress_hours: float,
) -> list[PlotFigure]:
    pair_specs: list[tuple[str, str]] = []
    for sensor in _ordered(trace_df["sensor"].astype(str).tolist()):
        sensor_metric = trace_df[
            (trace_df["sensor"].astype(str) == sensor) & (trace_df["metric"].astype(str) == selected_metric)
        ].copy()
        for sponge in _sponge_levels(sensor_metric, control_name=control_name):
            pair_specs.append((sensor, str(sponge)))
    if not pair_specs:
        return []
    rows, cols = best_subplot_grid(len(pair_specs))
    use_shared_axis_titles = selected_metric != "D_abs"
    policy = (
        _faceted_trace_figure_policy(rows=rows, cols=cols)
        if use_shared_axis_titles
        else _faceted_effect_trace_figure_policy(rows=rows, cols=cols)
    )
    stresses = _preferred_stresses(trace_df["stress_condition"], stress_order=stress_order)
    stress_color_map = _level_color_map(stresses, palette_book=palette_book)
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = _new_trace_figure(rows=rows, cols=cols, policy=policy, fig_kwargs=fig_kwargs)
        axes_flat = axes.ravel()
        legend_handles: dict[str, object] = {}
        for axis, (sensor, sponge) in zip(axes_flat, pair_specs, strict=False):
            pair_metric = trace_df[
                (trace_df["sensor"].astype(str) == sensor)
                & (trace_df["sponge"].astype(str) == sponge)
                & (trace_df["metric"].astype(str) == selected_metric)
            ].copy()
            if pair_metric.empty:
                axis.set_visible(False)
                continue
            has_iptg, iptg_levels = _trace_series_style(pair_metric)
            _plot_trace_panel_groups(
                ax=axis,
                frame=pair_metric,
                full_trace=full_trace[full_trace["sensor"].astype(str) == sensor].copy(),
                metric=selected_metric,
                grouped_levels=stresses,
                group_column="stress_condition",
                color_map=stress_color_map,
                has_iptg=has_iptg,
                iptg_levels=iptg_levels,
                legend_handles=legend_handles,
                fixed_sponge=sponge,
                iptg_labeler=lambda stress, iptg: f"{stress}, {iptg}",
            )
            _decorate_trace_axis(
                axis,
                metric=selected_metric,
                only_control=False,
                trace=pair_metric,
                stress_condition=None,
            )
            axis.set_title(f"{sensor} · {sponge}", pad=6, fontweight="normal", fontsize=10.6)
            axis.tick_params(axis="x", labelsize=8.6)
            axis.tick_params(axis="y", labelsize=8.6)
            _set_trace_axis_box_aspect(axis)
        for axis in axes_flat[len(pair_specs) :]:
            axis.set_visible(False)
        bounds = _trace_axis_bounds(
            metric_df=trace_df[trace_df["metric"].astype(str) == selected_metric].copy(),
            metric=selected_metric,
            only_control=False,
            full_trace=full_trace,
            display_post_stress_hours=display_post_stress_hours,
        )
        if bounds is not None:
            y_limits, display_bounds = bounds
            metric_ylabel = retron_presentation.compact_metric_axis_label(
                selected_metric,
                metric_label_map=metric_label_map,
            )
            for idx, axis in enumerate(axes_flat):
                if not axis.get_visible():
                    continue
                _apply_trace_axis_bounds(axis, y_limits=y_limits, display_bounds=display_bounds)
                axis.tick_params(labelbottom=True, labelleft=True)
                if use_shared_axis_titles:
                    axis.set_ylabel("")
                    axis.set_xlabel("")
                    continue
                axis.set_xlabel("Time from stress addition (h)", fontsize=10.2, labelpad=0.8)
                axis.set_ylabel(metric_ylabel if idx % cols == 0 else "", fontsize=11.2, labelpad=3.0)
        _finalize_trace_figure(
            fig,
            legend_handles=legend_handles,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            sensor="all sensors",
            subtitle=retron_presentation.trace_text_spec(selected_metric).figure_subtitle(),
            include_shared_xlabel=use_shared_axis_titles,
        )
        if use_shared_axis_titles:
            fig.supylabel(
                retron_presentation.metric_axis_label(selected_metric, metric_label_map=metric_label_map),
                x=0.04,
                fontsize=13.2,
            )
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
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
                    ax.set_ylabel(
                        retron_presentation.metric_axis_label(metric, metric_label_map=metric_label_map),
                        fontsize=13,
                    )
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
        trace_subtitle = (
            retron_presentation.trace_text_spec(selected_metrics[0]).figure_subtitle()
            if len(selected_metrics) == 1
            else ""
        )
        _finalize_trace_figure(
            fig,
            legend_handles=legend_handles,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            sensor=sensor,
            subtitle=trace_subtitle,
        )
        return emit_plot_figure(
            fig=fig,
            filename=f"{filename or _slug(title)}__sensor={_slug(sensor)}",
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _plot_control_trace_compact(
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
    display_post_stress_hours: float,
) -> list[PlotFigure]:
    column_specs = [(stress, metric) for stress in stresses for metric in selected_metrics]
    if not column_specs:
        return []
    figsize = fig_kwargs.get("figsize", (max(7.8, 2.55 * len(column_specs)), 3.15))
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = plt.subplots(
            1,
            len(column_specs),
            figsize=figsize,
            constrained_layout=False,
            squeeze=False,
            sharex=True,
            sharey=False,
        )
        axes_flat = axes.ravel()
        legend_handles: dict[str, object] = {}
        control_name = str(sensor_df["sponge"].astype(str).iloc[0]) if not sensor_df.empty else "tetO"
        for axis, (stress, metric) in zip(axes_flat, column_specs, strict=False):
            metric_df = sensor_df[
                (sensor_df["metric"].astype(str) == str(metric))
                & (sensor_df["stress_condition"].astype(str) == str(stress))
            ].copy()
            if metric_df.empty:
                axis.set_visible(False)
                continue
            color_map = _level_color_map([control_name], palette_book=palette_book)
            _plot_trace_panel_groups(
                ax=axis,
                frame=metric_df,
                full_trace=sensor_full_trace,
                metric=str(metric),
                grouped_levels=[control_name],
                group_column="sponge",
                color_map=color_map,
                has_iptg=True,
                iptg_levels=_trace_iptg_levels(metric_df),
                legend_handles=legend_handles,
                fixed_stress_condition=str(stress),
                iptg_labeler=lambda sponge, iptg: f"{iptg}",
            )
            _decorate_trace_axis(
                axis,
                metric=str(metric),
                only_control=True,
                trace=sensor_full_trace,
                stress_condition=str(stress),
            )
            metric_title = {
                "R": "Reporter ratio",
                "mu": "Growth rate",
            }.get(str(metric), retron_presentation.metric_axis_label(str(metric), metric_label_map=metric_label_map))
            axis.set_title(f"{str(stress)} · {metric_title}", pad=6, fontsize=10, fontweight="normal")
            axis.set_ylabel(
                retron_presentation.metric_axis_label(str(metric), metric_label_map=metric_label_map),
                fontsize=10.5,
            )
            axis.set_xlabel("Time from stress addition (h)", fontsize=9.2)
            axis.tick_params(axis="x", labelsize=8)
            axis.tick_params(axis="y", labelsize=8)
            _set_trace_axis_box_aspect(axis)
            bounds = _trace_axis_bounds(
                metric_df=sensor_df[sensor_df["metric"].astype(str) == str(metric)].copy(),
                metric=str(metric),
                only_control=True,
                full_trace=sensor_full_trace,
                display_post_stress_hours=display_post_stress_hours,
            )
            if bounds is not None:
                y_limits, display_bounds = bounds
                _apply_trace_axis_bounds(axis, y_limits=y_limits, display_bounds=display_bounds)
        for axis in axes_flat[len(column_specs) :]:
            axis.set_visible(False)
        fig.suptitle(f"{title} · {sensor}", y=0.98, x=0.5, ha="center", fontweight="normal", fontsize=13)
        if legend_handles:
            fig.legend(
                legend_handles.values(),
                legend_handles.keys(),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(2, len(legend_handles)),
                frameon=False,
                title=None,
            )
        fig.subplots_adjust(top=0.80, bottom=0.23, left=0.08, right=0.98, wspace=0.30)
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
    request = _RetronSummaryPlotRequest(
        summary=summary,
        trace=trace,
        output_dir=output_dir,
        title=title,
        filename=filename,
        palette_book=palette_book,
        control_name=control_name,
        no_stress_label=no_stress_label,
        relevant_only=relevant_only,
        metric=metric,
        state_order=state_order,
        burden_metric=burden_metric,
        fig_kwargs=fig_kwargs or {},
    )
    return _summary_view_renderer(view)(request)


def _summary_view_renderer(view: str) -> Callable[[_RetronSummaryPlotRequest], list[PlotFigure]]:
    try:
        return SUMMARY_VIEW_RENDERERS[str(view)]
    except KeyError as exc:
        raise ValueError(f"retron_sponge_summary: unsupported view {view!r}") from exc


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


def _trace_window_summary_frame(
    trace: pd.DataFrame,
    *,
    metric: str,
    stress_order: Sequence[str],
    positive_only: bool = False,
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
        if positive_only:
            values = np.maximum(values, 0.0)
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
    summary_metric = {"D": "O_AUC", "D_abs": "O_abs_AUC"}.get(str(metric))
    summary_trace_metric = {"D": "O", "D_abs": "O_abs"}.get(str(metric), str(metric))
    if not summary_metric:
        return
    summary = _trace_window_summary_frame(
        trace,
        metric=summary_trace_metric,
        stress_order=stress_order,
        positive_only=True,
    )
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
