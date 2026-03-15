"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/time_series.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pydantic import Field

from reader.core.plot_sinks import PlotFigure
from reader.core.registry import PluginConfig
from reader.core.workbench import PluginSemantics
from reader.plugins.plot._shared import FigurePlotPlugin, PlotPartitionCfg, resolve_plot_partition_cfg


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
    legend_loc: str = "upper left"
    show_replicates: bool = False
    filename: str | None = None


class TimeSeriesPlot(FigurePlotPlugin):
    key = "time_series"
    category = "plot"
    semantics = PluginSemantics(
        category="plot",
        domain="plate_reader",
        family="time_series",
        summary="Render grouped time-series plots from tidy plate-reader traces.",
        tags=("kinetics", "channels"),
    )
    ConfigModel = TimeSeriesCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1", "blanks?": "tidy.v1"}  # '?' is human hint; engine passes only present keys

    def render(self, ctx, inputs, cfg: TimeSeriesCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        blanks = inputs.get("blanks", df.iloc[0:0].copy())
        from reader.lib.microplates.time_series import plot_time_series  # noqa: PLC0415

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
            legend_loc=cfg.legend_loc,
            show_replicates=cfg.show_replicates,
            filename=cfg.filename,
        )
