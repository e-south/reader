"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/ts_and_snap.py

Two-panel plot (time series + snapshot barplot).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import Field

from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin, PlotPartitionCfg, resolve_plot_partition_cfg
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig


class TSAndSnapCfg(PluginConfig):
    # grouping
    partition: PlotPartitionCfg = Field(default_factory=PlotPartitionCfg)

    # time series (left)
    ts_x: str = "time"
    ts_channel: str
    ts_hue: str
    ts_time_window: list[float] | None = None
    ts_add_sheet_line: bool = False
    ts_sheet_line_kwargs: dict[str, Any] = Field(default_factory=dict)
    ts_mark_snap_time: bool = False
    ts_snap_line_kwargs: dict[str, Any] = Field(default_factory=dict)
    ts_log_transform: bool | list[str] = False
    ts_ci: float = 95.0
    ts_ci_alpha: float = 0.15
    ts_show_replicates: bool = False
    ts_legend_loc: str = "upper right"

    # snapshot (right)
    snap_x: str = "treatment"
    snap_channel: str | None = None
    snap_hue: str | None = None
    snap_time: float = Field(..., description="Snapshot time (hours); required for deterministic plotting.")
    snap_agg: Literal["mean", "median"] = "mean"
    snap_err: Literal["sem", "iqr", "none"] = "sem"
    snap_time_tolerance: float = 0.51
    snap_show_legend: bool = False
    snap_legend_loc: str = "upper right"

    # figure
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None


class TSAndSnapPlot(FigurePlotPlugin):
    ConfigModel = TSAndSnapCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    def render(self, ctx, inputs, cfg: TSAndSnapCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        from reader.domains.plate_reader.plots.ts_and_snap import plot_ts_and_snap  # noqa: PLC0415

        partition = resolve_plot_partition_cfg(ctx=ctx, partition=cfg.partition)

        return plot_ts_and_snap(
            df=df,
            output_dir=None,
            group_on=partition.group_by,
            pool_sets=partition.collection_items,
            pool_match=partition.match,
            # ts (left)
            ts_x=cfg.ts_x,
            ts_channel=cfg.ts_channel,
            ts_hue=cfg.ts_hue,
            ts_time_window=cfg.ts_time_window,
            ts_add_sheet_line=cfg.ts_add_sheet_line,
            ts_sheet_line_kwargs=cfg.ts_sheet_line_kwargs,
            ts_mark_snap_time=cfg.ts_mark_snap_time,
            ts_snap_line_kwargs=cfg.ts_snap_line_kwargs,
            ts_log_transform=cfg.ts_log_transform,
            ts_ci=cfg.ts_ci,
            ts_ci_alpha=cfg.ts_ci_alpha,
            ts_show_replicates=cfg.ts_show_replicates,
            ts_legend_loc=cfg.ts_legend_loc,
            # snap (right)
            snap_x=cfg.snap_x,
            snap_channel=cfg.snap_channel,
            snap_hue=cfg.snap_hue,
            snap_time=cfg.snap_time,
            snap_agg=cfg.snap_agg,
            snap_err=cfg.snap_err,
            snap_time_tolerance=cfg.snap_time_tolerance,
            snap_show_legend=cfg.snap_show_legend,
            snap_legend_loc=cfg.snap_legend_loc,
            # fig
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
        )
