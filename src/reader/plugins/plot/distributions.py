"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/distributions.py

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


class DistributionsCfg(PluginConfig):
    # what to draw
    channels: list[str]
    # modern grouping
    partition: PlotPartitionCfg = Field(default_factory=lambda: PlotPartitionCfg(by="design_id"))
    # layout
    panel_by: Literal["channel", "group"] = "channel"  # default: per-channel panels
    hue: str | None = None
    legend_loc: Literal["upper left", "upper right", "lower left", "lower right", "center", "best"] = "upper left"
    # style/output
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None


class DistributionsPlot(FigurePlotPlugin):
    key = "distributions"
    category = "plot"
    semantics = PluginSemantics(
        category="plot",
        domain="plate_reader",
        family="distribution",
        summary="Render channel-wise distribution plots from tidy measurements.",
        tags=("density", "qc"),
    )
    ConfigModel = DistributionsCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        # blanks is optional; if absent we’ll draw only data fills
        return {"df": "tidy.v1", "blanks?": "tidy.v1"}

    def render(self, ctx, inputs, cfg: DistributionsCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        blanks: pd.DataFrame = inputs.get("blanks", df.iloc[0:0])
        from reader.lib.microplates.distributions import plot_distributions  # noqa: PLC0415

        partition = resolve_plot_partition_cfg(ctx=ctx, partition=cfg.partition)

        return plot_distributions(
            df=df,
            blanks=blanks,
            output_dir=None,
            channels=cfg.channels,
            group_on=partition.group_by,
            pool_sets=partition.collection_items,
            pool_match=partition.match,  # type: ignore[arg-type]
            panel_by=cfg.panel_by,
            hue=cfg.hue,
            legend_loc=cfg.legend_loc,
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
        )
