from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

from reader.plotting.sinks import PlotFigure
from reader.plotting.style import use_style

from .. import _retron_sponge_presentation as retron_presentation
from ..common import annotate_points_smart, emit_plot_figure, warn_if_empty
from .shared import (
    _finalize_summary_figure,
    _first_non_null,
    _level_color_map,
    _new_summary_grid_figure,
    _ordered,
    _RetronSummaryPlotRequest,
    _slug,
    _SummaryFigurePolicy,
    _SummarySubplotPolicy,
    _wrap_hyphenated_label,
)


@dataclass(frozen=True)
class _ParetoPointPayload:
    x: float
    y: float
    size: float
    color: str
    label: str


@dataclass(frozen=True)
class _ParetoFigurePayload:
    points: tuple[_ParetoPointPayload, ...]
    family_levels: tuple[str, ...]
    color_map: dict[str, str]
    x_label: str
    y_label: str


@dataclass(frozen=True)
class _ParetoAxisPolicy:
    zero_line_color: str
    zero_line_linewidth: float
    zero_line_linestyle: str
    xlabel_fontsize: float
    ylabel_fontsize: float
    tick_size: float
    legend_loc: str
    legend_bbox_to_anchor: tuple[float, float]
    legend_borderaxespad: float
    box_aspect: float


_PARETO_AXIS_POLICY = _ParetoAxisPolicy(
    zero_line_color="#777777",
    zero_line_linewidth=1.0,
    zero_line_linestyle=":",
    xlabel_fontsize=11.0,
    ylabel_fontsize=11.0,
    tick_size=7.0,
    legend_loc="center left",
    legend_bbox_to_anchor=(1.01, 0.5),
    legend_borderaxespad=0.0,
    box_aspect=1.0,
)


def render_pareto_view(request: _RetronSummaryPlotRequest) -> list[PlotFigure]:
    return _plot_retron_pareto(
        summary=request.summary,
        trace=request.trace,
        output_dir=request.output_dir,
        title=request.title,
        filename=request.filename,
        palette_book=request.palette_book,
        control_name=request.control_name,
        metric=str(request.metric or "S_abs_AUC"),
        burden_metric=request.burden_metric,
        fig_kwargs=request.fig_kwargs,
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


def _plot_retron_pareto(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir,
    title: str,
    filename: str | None,
    palette_book,
    control_name: str,
    metric: str,
    burden_metric: str,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    relevant = _pareto_relevant_frame(summary, control_name=control_name)
    if warn_if_empty(relevant, where="retron_pareto", detail="after on-target filtering"):
        return []
    payload = _pareto_figure_payload(
        summary=summary,
        relevant=relevant,
        control_name=control_name,
        metric=metric,
        burden_metric=burden_metric,
        palette_book=palette_book,
    )
    if not payload.points:
        warn_if_empty(pd.DataFrame(), where="retron_pareto", detail="after aggregation")
        return []
    policy = _pareto_figure_policy()
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = _new_summary_grid_figure(
            rows=1,
            cols=1,
            policy=policy,
            fig_kwargs=fig_kwargs,
        )
        ax = axes[0][0]
        _render_pareto_points(ax, payload=payload)
        _decorate_pareto_axis(ax, payload=payload, policy=_PARETO_AXIS_POLICY)
        _apply_pareto_legend(ax, payload=payload, policy=_PARETO_AXIS_POLICY)
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


def _pareto_figure_payload(
    *,
    summary: pd.DataFrame,
    relevant: pd.DataFrame,
    control_name: str,
    metric: str,
    burden_metric: str,
    palette_book,
) -> _ParetoFigurePayload:
    table = _pareto_summary_frame(
        summary=summary,
        relevant=relevant,
        control_name=control_name,
        metric=metric,
        burden_metric=burden_metric,
    )
    family_levels, color_map = _pareto_family_colors(table, palette_book=palette_book)
    sizes = _pareto_marker_sizes(table).to_numpy(dtype=float)
    points = tuple(
        _ParetoPointPayload(
            x=float(row.on_target),
            y=float(row.burden),
            size=float(size),
            color=str(color_map.get(str(row.family), "#4c72b0")),
            label=_wrap_hyphenated_label(str(row.sponge), max_parts_per_line=2),
        )
        for (_, row), size in zip(table.iterrows(), sizes, strict=False)
    )
    return _ParetoFigurePayload(
        points=points,
        family_levels=tuple(family_levels),
        color_map=color_map,
        x_label=f"Mean on-target effect across relevant sensors ({metric})",
        y_label=retron_presentation.burden_axis_label(burden_metric),
    )


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
    palette_book,
) -> tuple[list[str], dict[str, str]]:
    family_levels = _ordered(
        table["family"].fillna("other").astype(str).tolist(),
        preferred=("mono", "bi", "tri", "quad", "other"),
    )
    return family_levels, _level_color_map(family_levels, palette_book=palette_book)


def _pareto_marker_sizes(table: pd.DataFrame) -> pd.Series:
    return 80.0 + 240.0 * table["leakiness"].abs().fillna(0.0)


def _render_pareto_points(
    ax,
    *,
    payload: _ParetoFigurePayload,
) -> None:
    if not payload.points:
        return
    ax.scatter(
        [point.x for point in payload.points],
        [point.y for point in payload.points],
        s=[point.size for point in payload.points],
        c=[point.color for point in payload.points],
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    annotate_points_smart(
        ax=ax,
        points=[(point.x, point.y) for point in payload.points],
        labels=[point.label for point in payload.points],
    )


def _decorate_pareto_axis(
    ax,
    *,
    payload: _ParetoFigurePayload,
    policy: _ParetoAxisPolicy,
) -> None:
    ax.axvline(
        0.0,
        color=policy.zero_line_color,
        linewidth=policy.zero_line_linewidth,
        linestyle=policy.zero_line_linestyle,
    )
    ax.axhline(
        0.0,
        color=policy.zero_line_color,
        linewidth=policy.zero_line_linewidth,
        linestyle=policy.zero_line_linestyle,
    )
    ax.set_xlabel(payload.x_label, fontsize=policy.xlabel_fontsize)
    ax.set_ylabel(payload.y_label, fontsize=policy.ylabel_fontsize)
    ax.tick_params(axis="both", labelsize=policy.tick_size)
    with suppress(Exception):
        ax.set_box_aspect(policy.box_aspect)


def _apply_pareto_legend(
    ax,
    *,
    payload: _ParetoFigurePayload,
    policy: _ParetoAxisPolicy,
) -> None:
    legend_handles = _pareto_legend_handles(payload)
    if not legend_handles:
        return
    ax.legend(
        handles=legend_handles,
        frameon=False,
        title=None,
        loc=policy.legend_loc,
        bbox_to_anchor=policy.legend_bbox_to_anchor,
        borderaxespad=policy.legend_borderaxespad,
    )


def _pareto_legend_handles(
    payload: _ParetoFigurePayload,
) -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=level,
            markerfacecolor=payload.color_map[level],
            markersize=8,
        )
        for level in payload.family_levels
    ]
