"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/time_series.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin, PlotPartitionCfg, resolve_plot_partition_cfg
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig


class TimeSeriesCfg(PluginConfig):
    x: str = "time"
    y: list[str] | None = None
    hue: str = "treatment"
    partition: PlotPartitionCfg = Field(default_factory=PlotPartitionCfg)
    fig: dict[str, Any] = Field(default_factory=dict)
    channels: list[str] | None = None
    add_sheet_line: bool = False
    sheet_line_kwargs: dict[str, Any] = Field(default_factory=dict)
    log_transform: bool | list[str] = False
    time_window: list[float] | None = None
    ci: float = 95.0
    ci_alpha: float = 0.15
    ci_boot: int = Field(default=100, ge=1)
    ci_seed: int = 0
    legend_loc: str = "upper left"
    show_replicates: bool = False
    filename: str | None = None


class TimeSeriesPlot(FigurePlotPlugin):
    ConfigModel = TimeSeriesCfg

    @classmethod
    def input_ports(cls):
        return {
            "df": dataframe_input("df", "tidy.v1"),
            "blanks": dataframe_input("blanks", "tidy.v1", optional=True),
        }

    def render(self, ctx, inputs, cfg: TimeSeriesCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        blanks = inputs.get("blanks", df.iloc[0:0].copy())
        from reader.domains.plate_reader.plots.time_series import plot_time_series  # noqa: PLC0415

        partition = resolve_plot_partition_cfg(ctx=ctx, partition=cfg.partition)

        return plot_time_series(
            df=df,
            blanks=blanks,
            output_dir=None,
            x=cfg.x,
            y=cfg.y,
            hue=cfg.hue,
            channels=cfg.channels,
            subplots=None,
            group_on=partition.group_by,
            pool_sets=partition.collection_items,
            pool_match=partition.match,
            fig_kwargs=cfg.fig,
            add_sheet_line=cfg.add_sheet_line,
            sheet_line_kwargs=cfg.sheet_line_kwargs,
            log_transform=cfg.log_transform,
            time_window=cfg.time_window,
            palette_book=ctx.palette_book,
            ci=cfg.ci,
            ci_alpha=cfg.ci_alpha,
            ci_boot=cfg.ci_boot,
            ci_seed=cfg.ci_seed,
            legend_loc=cfg.legend_loc,
            show_replicates=cfg.show_replicates,
            filename=cfg.filename,
        )
