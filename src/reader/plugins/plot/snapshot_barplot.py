"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/snapshot_barplot.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from reader.core.plot_sinks import PlotFigure
from reader.core.registry import PluginConfig
from reader.core.workbench import PluginSemantics
from reader.plugins.plot._shared import FigurePlotPlugin, PlotPartitionCfg, resolve_plot_partition_cfg


class SnapshotBarCfg(PluginConfig):
    x: str
    y: list[str] | str
    hue: str | None = None
    partition: PlotPartitionCfg = Field(default_factory=PlotPartitionCfg)
    time: float = Field(..., description="Snapshot time (hours); required for deterministic plotting.")
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None
    agg: str = "mean"  # median|mean
    err: str = "sem"  # iqr|sem|none
    time_tolerance: float = 0.51
    panel_by: Literal["channel", "x", "group"] = "channel"
    channel_select: str | None = None
    file_by: Literal["auto", "channel"] = "auto"
    show_legend: bool = False
    legend_loc: str = "upper right"


class SnapshotBarplot(FigurePlotPlugin):
    key = "snapshot_barplot"
    category = "plot"
    semantics = PluginSemantics(
        category="plot",
        domain="plate_reader",
        family="snapshot_bar",
        summary="Render grouped snapshot barplots at a selected timepoint.",
        tags=("snapshot", "bars"),
    )
    ConfigModel = SnapshotBarCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    def render(self, ctx, inputs, cfg: SnapshotBarCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        from reader.lib.microplates.snapshot_barplot import plot_snapshot_barplot  # noqa: PLC0415

        partition = resolve_plot_partition_cfg(ctx=ctx, partition=cfg.partition)

        return plot_snapshot_barplot(
            df=df,
            output_dir=None,
            x=cfg.x,
            y=cfg.y,
            hue=cfg.hue,
            group_on=partition.group_by,
            pool_sets=partition.collection_items,
            time=cfg.time,
            pool_match=partition.match,  # type: ignore
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
            agg=cfg.agg,
            err=cfg.err,
            time_tolerance=cfg.time_tolerance,
            panel_by=cfg.panel_by,
            channel_select=cfg.channel_select,
            file_by=cfg.file_by,
            show_legend=cfg.show_legend,
            legend_loc=cfg.legend_loc,
        )
