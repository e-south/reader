"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/logic_symmetry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pydantic import Field

from reader.core.plot_sinks import PlotFigure, normalize_plot_figures, save_plot_figures
from reader.core.registry import Plugin, PluginConfig


class LogicSymCfg(PluginConfig):
    response_channel: str
    design_by: list[str] = Field(default_factory=lambda: ["design_id"])
    batch_col: str = "batch"
    treatment_map: dict[str, str]
    treatment_case_sensitive: bool = True
    aggregation: dict[str, Any] = Field(default_factory=dict)
    encodings: dict[str, Any] = Field(default_factory=dict)
    ideals_overlay: dict[str, Any] = Field(default_factory=dict)
    visuals: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    prep: dict[str, Any] | None = None
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None


class LogicSymmetryPlot(Plugin):
    key = "logic_symmetry"
    category = "plot"
    ConfigModel = LogicSymCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        # tidy+map helps ensure design_by & batch present; for leniency accept tidy.v1 in earlier range
        return {"df": "tidy+map.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def render(self, ctx, inputs, cfg: LogicSymCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        from reader.lib.logic_symmetry import plot_logic_symmetry

        result = plot_logic_symmetry(
            df=df,
            blanks=df.iloc[0:0],
            output_dir=None,
            response_channel=cfg.response_channel,
            design_by=cfg.design_by,
            batch_col=cfg.batch_col,
            treatment_map=cfg.treatment_map,
            treatment_case_sensitive=cfg.treatment_case_sensitive,
            aggregation=cfg.aggregation,
            encodings=cfg.encodings,
            ideals_overlay=cfg.ideals_overlay,
            visuals=cfg.visuals,
            output=cfg.output,
            prep=cfg.prep,
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
        )
        formats = [str(x).lower() for x in (cfg.output or {}).get("format", ["pdf"])]
        dpi = (cfg.output or {}).get("dpi", 300)
        base = cfg.filename or "logic_symmetry"
        return [PlotFigure(fig=result.fig, filename=base, ext=ext, dpi=dpi) for ext in formats]

    def run(self, ctx, inputs, cfg: LogicSymCfg):
        figures = normalize_plot_figures(self.render(ctx, inputs, cfg), where=f"plot/{self.key}")
        saved = save_plot_figures(figures, ctx.plots_dir)
        return {"files": [str(p) for p in saved] if saved else None}
