from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns

from reader.plotting.sinks import PlotFigure
from reader.plotting.style import use_style

from .. import _retron_sponge_presentation as retron_presentation
from ..common import emit_plot_figure, shared_numeric_limits, warn_if_empty
from .shared import (
    _finalize_summary_figure,
    _new_summary_grid_figure,
    _pivot_summary,
    _require_relevant_sensor_pair,
    _RetronSummaryPlotRequest,
    _set_axis_title,
    _slug,
    _SummaryFigurePolicy,
    _SummarySubplotPolicy,
    _wrap_hyphenated_label,
)


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
class _LibraryHeatmapFigurePayload:
    panel_payloads: tuple[_HeatmapPanelPayload, ...]
    panel_limits: dict[str, tuple[float, float]]
    max_rows: int
    max_cols: int


@dataclass(frozen=True)
class _LibraryHeatmapAxisPolicy:
    cmap: str
    center: float
    annotation_format: str
    annotation_fontsize: float
    linewidths: float
    linecolor: str
    title_pad: float
    xtick_labelsize: float
    xtick_pad: float
    ytick_labelsize: float
    wrap_xtick_parts_per_line: int
    no_data_text: str


_LIBRARY_HEATMAP_PANELS: tuple[_HeatmapPanelSpec, ...] = (
    _HeatmapPanelSpec(
        metric="S_abs_AUC",
        title="Total effect",
        formula="S_abs_AUC = O_abs_AUC / |G_sensor|",
        scale_group="scaled",
    ),
    _HeatmapPanelSpec(
        metric="S_AUC",
        title="Post-stress",
        formula="S_AUC = O_AUC / |G_sensor|",
        scale_group="scaled",
    ),
    _HeatmapPanelSpec(
        metric="P_pre",
        title="Preload",
        formula="P_pre = delta_IPTG[R_pre - R_pre,tetO]",
        scale_group="delta",
    ),
)

_LIBRARY_HEATMAP_AXIS_POLICY = _LibraryHeatmapAxisPolicy(
    cmap="vlag",
    center=0.0,
    annotation_format=".2f",
    annotation_fontsize=8.5,
    linewidths=0.4,
    linecolor="#f0f0f0",
    title_pad=8.0,
    xtick_labelsize=9.0,
    xtick_pad=1.0,
    ytick_labelsize=10.0,
    wrap_xtick_parts_per_line=2,
    no_data_text="No data",
)


def render_library_heatmap_view(request: _RetronSummaryPlotRequest) -> list[PlotFigure]:
    return _plot_retron_library_heatmaps(
        summary=request.summary,
        trace=request.trace,
        output_dir=request.output_dir,
        title=request.title,
        filename=request.filename,
        control_name=request.control_name,
        relevant_only=request.relevant_only,
        fig_kwargs=request.fig_kwargs,
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


def _plot_retron_library_heatmaps(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir,
    title: str,
    filename: str | None,
    control_name: str,
    relevant_only: bool,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    df = summary[summary["sponge"].astype(str) != str(control_name)].copy()
    if relevant_only:
        _require_relevant_sensor_pair(df, where="retron_library_heatmaps")
        df = df[df["relevant_sensor_pair"].fillna(False)]
    if warn_if_empty(df, where="retron_library_heatmaps", detail="after control/on-target filters"):
        return []
    figure_payload = _library_heatmap_figure_payload(df)
    policy = _library_heatmap_figure_policy(max_rows=figure_payload.max_rows, max_cols=figure_payload.max_cols)
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = _new_summary_grid_figure(
            rows=1,
            cols=len(figure_payload.panel_payloads),
            policy=policy,
            fig_kwargs=fig_kwargs,
            sharey=True,
        )
        for panel_index, (ax, payload) in enumerate(zip(axes.ravel(), figure_payload.panel_payloads, strict=False)):
            _plot_library_heatmap_panel(
                ax,
                payload=payload,
                panel_limits=figure_payload.panel_limits,
                panel_index=panel_index,
            )
        _finalize_summary_figure(
            fig,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            subtitle=retron_presentation.render_summary_text(
                retron_presentation.LIBRARY_HEATMAP_TEXT_SPEC,
                trace=trace,
            ),
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


def _library_heatmap_figure_payload(df: pd.DataFrame) -> _LibraryHeatmapFigurePayload:
    panel_payloads = tuple(_library_heatmap_panel_payloads(df))
    return _LibraryHeatmapFigurePayload(
        panel_payloads=panel_payloads,
        panel_limits=_library_heatmap_limits(panel_payloads),
        max_rows=max(max(1, len(payload.pivot.index)) for payload in panel_payloads),
        max_cols=max(max(1, len(payload.pivot.columns)) for payload in panel_payloads),
    )


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
        _render_empty_library_heatmap_panel(ax, policy=_LIBRARY_HEATMAP_AXIS_POLICY)
        return
    _render_library_heatmap(ax, payload=payload, panel_limits=panel_limits, policy=_LIBRARY_HEATMAP_AXIS_POLICY)
    _decorate_library_heatmap_axis(
        ax,
        payload=payload,
        panel_index=panel_index,
        policy=_LIBRARY_HEATMAP_AXIS_POLICY,
    )


def _render_empty_library_heatmap_panel(ax, *, policy: _LibraryHeatmapAxisPolicy) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, policy.no_data_text, ha="center", va="center")


def _render_library_heatmap(
    ax,
    *,
    payload: _HeatmapPanelPayload,
    panel_limits: Mapping[str, tuple[float, float]],
    policy: _LibraryHeatmapAxisPolicy,
) -> None:
    sns.heatmap(
        payload.pivot,
        ax=ax,
        cmap=policy.cmap,
        center=policy.center,
        annot=True,
        fmt=policy.annotation_format,
        cbar=False,
        square=False,
        linewidths=policy.linewidths,
        linecolor=policy.linecolor,
        annot_kws={"fontsize": policy.annotation_fontsize},
        vmin=panel_limits.get(payload.spec.scale_group, (None, None))[0],
        vmax=panel_limits.get(payload.spec.scale_group, (None, None))[1],
    )


def _decorate_library_heatmap_axis(
    ax,
    *,
    payload: _HeatmapPanelPayload,
    panel_index: int,
    policy: _LibraryHeatmapAxisPolicy,
) -> None:
    _set_axis_title(ax, f"{payload.spec.title}\n{payload.spec.formula}", pad=policy.title_pad)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        [
            _wrap_hyphenated_label(str(label), max_parts_per_line=policy.wrap_xtick_parts_per_line)
            for label in payload.pivot.columns
        ]
    )
    ax.tick_params(axis="x", labelrotation=0, labelsize=policy.xtick_labelsize, pad=policy.xtick_pad)
    for label in ax.get_xticklabels():
        label.set_ha("center")
    if panel_index > 0:
        ax.tick_params(axis="y", labelleft=False)
        return
    ax.tick_params(axis="y", labelrotation=0, labelsize=policy.ytick_labelsize)
