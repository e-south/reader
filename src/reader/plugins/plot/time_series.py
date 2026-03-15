"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/time_series.py

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


class TimeSeriesCfg(PluginConfig):
    x: str = "time"
    y: list[str] | None = None
    hue: str = "treatment"
    group_on: str | None = None
    pool_sets: str | list[dict[str, list[str]]] | None = None
    pool_match: Literal["exact", "contains", "startswith", "endswith", "regex"] = "exact"
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


class TimeSeriesPlot(Plugin):
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

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def render(self, ctx, inputs, cfg: TimeSeriesCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        blanks = inputs.get("blanks", df.iloc[0:0].copy())
        from reader.lib.microplates.time_series import plot_time_series  # noqa: PLC0415

        resolved_pools = resolve_pool_sets_arg(pool_sets=cfg.pool_sets, group_on=cfg.group_on, groups=ctx.groups)

        return plot_time_series(
            df=df,
            blanks=blanks,
            output_dir=None,
            x=cfg.x,
            y=cfg.y,
            hue=cfg.hue,
            channels=cfg.channels,
            subplots=None,
            group_on=cfg.group_on,
            pool_sets=resolved_pools,
            pool_match=cfg.pool_match,
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

    def run(self, ctx, inputs, cfg: TimeSeriesCfg):
        return save_rendered_figures(ctx=ctx, figures=self.render(ctx, inputs, cfg), plot_key=self.key)
