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
from reader.core.registry import Plugin, PluginConfig
from reader.core.semantics import resolve_pool_sets_arg
from reader.core.workbench import PluginSemantics
from reader.plugins.plot._shared import save_rendered_figures


class SnapshotBarCfg(PluginConfig):
    x: str
    y: list[str] | str
    hue: str | None = None
    group_on: str | None = None
    pool_sets: str | list[dict[str, list[str]]] | None = None
    time: float = Field(..., description="Snapshot time (hours); required for deterministic plotting.")
    pool_match: Literal["exact", "contains", "startswith", "endswith", "regex"] = "exact"
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


class SnapshotBarplot(Plugin):
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

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def render(self, ctx, inputs, cfg: SnapshotBarCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        from reader.lib.microplates.snapshot_barplot import plot_snapshot_barplot  # noqa: PLC0415

        resolved_pools = resolve_pool_sets_arg(pool_sets=cfg.pool_sets, group_on=cfg.group_on, groups=ctx.groups)

        return plot_snapshot_barplot(
            df=df,
            output_dir=None,
            x=cfg.x,
            y=cfg.y,
            hue=cfg.hue,
            group_on=cfg.group_on,
            pool_sets=resolved_pools,
            time=cfg.time,
            pool_match=cfg.pool_match,  # type: ignore
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

    def run(self, ctx, inputs, cfg: SnapshotBarCfg):
        return save_rendered_figures(ctx=ctx, figures=self.render(ctx, inputs, cfg), plot_key=self.key)
