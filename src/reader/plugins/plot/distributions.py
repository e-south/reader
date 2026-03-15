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
from reader.core.registry import Plugin, PluginConfig
from reader.core.semantics import resolve_pool_sets_arg
from reader.core.workbench import PluginSemantics
from reader.plugins.plot._shared import save_rendered_figures


class DistributionsCfg(PluginConfig):
    # what to draw
    channels: list[str]
    # modern grouping
    group_on: str | None = "design_id"
    pool_sets: str | list[str] | list[dict[str, list[str]]] | None = None
    pool_match: Literal["exact", "contains", "startswith", "endswith", "regex"] = "exact"
    # layout
    panel_by: Literal["channel", "group"] = "channel"  # default: per-channel panels
    hue: str | None = None
    legend_loc: Literal["upper left", "upper right", "lower left", "lower right", "center", "best"] = "upper left"
    # style/output
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None


class DistributionsPlot(Plugin):
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

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def render(self, ctx, inputs, cfg: DistributionsCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        blanks: pd.DataFrame = inputs.get("blanks", df.iloc[0:0])
        from reader.lib.microplates.distributions import plot_distributions  # noqa: PLC0415

        resolved_pools = resolve_pool_sets_arg(
            pool_sets=cfg.pool_sets,
            group_on=cfg.group_on,
            groups=ctx.groups,
            allow_reference_lists=True,
        )

        return plot_distributions(
            df=df,
            blanks=blanks,
            output_dir=None,
            channels=cfg.channels,
            group_on=cfg.group_on,
            pool_sets=resolved_pools,
            pool_match=cfg.pool_match,  # type: ignore[arg-type]
            panel_by=cfg.panel_by,
            hue=cfg.hue,
            legend_loc=cfg.legend_loc,
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
        )

    def run(self, ctx, inputs, cfg: DistributionsCfg):
        return save_rendered_figures(ctx=ctx, figures=self.render(ctx, inputs, cfg), plot_key=self.key)
