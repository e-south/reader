from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reader.plotting.sinks import PlotFigure
from reader.plotting.style import use_style

from .. import _retron_sponge_presentation as retron_presentation
from ..common import emit_plot_figure, shared_numeric_limits, warn_if_empty
from .shared import (
    _finalize_summary_figure,
    _level_color_map,
    _new_summary_grid_figure,
    _ordered,
    _require_relevant_sensor_pair,
    _RetronSummaryPlotRequest,
    _slug,
    _SummaryFigurePolicy,
    _SummarySubplotPolicy,
    _wrap_hyphenated_label,
)


@dataclass(frozen=True)
class _StressModulationChartPayload:
    row_labels: tuple[str, ...]
    base_positions: np.ndarray
    bar_height: float
    sample_values: np.ndarray
    control_values: np.ndarray
    sample_mask: np.ndarray
    control_mask: np.ndarray
    sensor_labels: np.ndarray
    x_limits: tuple[float, float]


@dataclass(frozen=True)
class _StressModulationAxisPolicy:
    xlabel: str
    xlabel_fontsize: float
    tick_size: float
    grid_color: str
    grid_linewidth: float
    grid_alpha: float
    zero_line_color: str
    zero_line_linewidth: float
    zero_line_linestyle: str
    legend_loc: str
    legend_bbox_to_anchor: tuple[float, float]
    legend_borderaxespad: float


def render_stress_modulation_view(request: _RetronSummaryPlotRequest) -> list[PlotFigure]:
    return _plot_retron_stress_modulation(
        summary=request.summary,
        trace=request.trace,
        output_dir=request.output_dir,
        title=request.title,
        filename=request.filename,
        palette_book=request.palette_book,
        control_name=request.control_name,
        relevant_only=request.relevant_only,
        metric=str(request.metric or "M_AUC"),
        fig_kwargs=request.fig_kwargs,
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


def _stress_modulation_axis_policy(metric: str) -> _StressModulationAxisPolicy:
    return _StressModulationAxisPolicy(
        xlabel=retron_presentation.summary_metric_label(metric),
        xlabel_fontsize=11.0,
        tick_size=8.0,
        grid_color="#d9d9d9",
        grid_linewidth=0.6,
        grid_alpha=0.55,
        zero_line_color="#777777",
        zero_line_linewidth=1.0,
        zero_line_linestyle=":",
        legend_loc="center left",
        legend_bbox_to_anchor=(1.01, 0.5),
        legend_borderaxespad=0.0,
    )


def _plot_retron_stress_modulation(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir,
    title: str,
    filename: str | None,
    palette_book,
    control_name: str,
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
    chart_data = _stress_modulation_chart_payload(plot_df)
    axis_policy = _stress_modulation_axis_policy(metric)
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = _new_summary_grid_figure(
            rows=1,
            cols=1,
            policy=policy,
            fig_kwargs=fig_kwargs,
        )
        ax = axes[0][0]
        _plot_stress_modulation_bars(ax, plot_df=plot_df, chart_data=chart_data, sensor_colors=sensor_colors)
        _decorate_stress_modulation_axis(ax, chart_data=chart_data, policy=axis_policy)
        _finalize_summary_figure(
            fig,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            subtitle=retron_presentation.render_summary_text(
                retron_presentation.summary_metric_text_spec(metric),
                trace=trace,
            ),
        )
        _apply_stress_modulation_legend(ax, chart_data=chart_data, policy=axis_policy)
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _stress_modulation_chart_payload(plot_df: pd.DataFrame) -> _StressModulationChartPayload:
    sample_values = pd.to_numeric(plot_df["sample_value"], errors="coerce").to_numpy(dtype=float)
    control_values = pd.to_numeric(plot_df["control_value"], errors="coerce").to_numpy(dtype=float)
    combined = np.concatenate([sample_values[np.isfinite(sample_values)], control_values[np.isfinite(control_values)]])
    return _StressModulationChartPayload(
        row_labels=tuple(
            _stress_modulation_row_label(sensor=str(row.sensor), sponge=str(row.sponge))
            for row in plot_df.itertuples(index=False)
        ),
        base_positions=np.arange(len(plot_df), dtype=float),
        bar_height=0.34,
        sample_values=sample_values,
        control_values=control_values,
        sample_mask=np.isfinite(sample_values),
        control_mask=np.isfinite(control_values),
        sensor_labels=np.array(
            [str(sensor) for sensor in plot_df["sensor"].astype(str)],
            dtype=object,
        ),
        x_limits=shared_numeric_limits(
            combined if combined.size else np.array([0.0], dtype=float),
            center=0.0,
            pad_fraction=0.10,
            min_span=0.10,
        ),
    )


def _plot_stress_modulation_bars(
    ax,
    *,
    plot_df: pd.DataFrame,
    chart_data: _StressModulationChartPayload,
    sensor_colors: Mapping[str, str],
) -> None:
    base_positions = chart_data.base_positions
    bar_height = chart_data.bar_height
    sample_values = chart_data.sample_values
    control_values = chart_data.control_values
    sample_mask = chart_data.sample_mask
    control_mask = chart_data.control_mask
    sensor_labels = chart_data.sensor_labels
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


def _decorate_stress_modulation_axis(
    ax,
    *,
    chart_data: _StressModulationChartPayload,
    policy: _StressModulationAxisPolicy,
) -> None:
    ax.set_xlim(chart_data.x_limits)
    ax.axvline(
        0.0,
        color=policy.zero_line_color,
        linewidth=policy.zero_line_linewidth,
        linestyle=policy.zero_line_linestyle,
    )
    ax.set_xlabel(policy.xlabel, fontsize=policy.xlabel_fontsize)
    ax.set_ylabel("")
    ax.set_yticks(chart_data.base_positions)
    ax.set_yticklabels(chart_data.row_labels)
    ax.tick_params(axis="x", labelsize=policy.tick_size)
    ax.tick_params(axis="y", labelsize=policy.tick_size)
    ax.grid(
        axis="x",
        color=policy.grid_color,
        linewidth=policy.grid_linewidth,
        alpha=policy.grid_alpha,
    )


def _apply_stress_modulation_legend(
    ax,
    *,
    chart_data: _StressModulationChartPayload,
    policy: _StressModulationAxisPolicy,
) -> None:
    if not (chart_data.control_mask.any() or chart_data.sample_mask.any()):
        return
    ax.legend(
        frameon=False,
        title=None,
        loc=policy.legend_loc,
        bbox_to_anchor=policy.legend_bbox_to_anchor,
        borderaxespad=policy.legend_borderaxespad,
    )


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
