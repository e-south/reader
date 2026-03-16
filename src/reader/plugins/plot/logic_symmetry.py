"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/logic_symmetry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig


class LogicSymCfg(PluginConfig):
    response_channel: str
    design_by: list[str] = Field(default_factory=lambda: ["design_id"])
    batch_col: str = "batch"
    logic_map_ref: str
    aggregation: dict[str, Any] = Field(default_factory=dict)
    encodings: dict[str, Any] = Field(default_factory=dict)
    ideals_overlay: dict[str, Any] = Field(default_factory=dict)
    visuals: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    prep: dict[str, Any] | None = None
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None


class LogicSymmetryPlot(FigurePlotPlugin):
    ConfigModel = LogicSymCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "plate_reader.annotated.v1")}

    def render(self, ctx, inputs, cfg: LogicSymCfg) -> list[PlotFigure]:
        if ctx.experiment is None:
            raise ValueError("logic_symmetry requires experiment semantics in the run context")
        df: pd.DataFrame = inputs["df"]
        from reader.domains.logic.logic_symmetry import plot_logic_symmetry  # noqa: PLC0415

        logic_map = ctx.experiment.annotations.resolve_logic_map(ref=cfg.logic_map_ref)
        result = plot_logic_symmetry(
            df=df,
            blanks=df.iloc[0:0],
            output_dir=None,
            response_channel=cfg.response_channel,
            design_by=cfg.design_by,
            batch_col=cfg.batch_col,
            treatment_map=dict(logic_map.corners),
            treatment_case_sensitive=logic_map.case_sensitive,
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
