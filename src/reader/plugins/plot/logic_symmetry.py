"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/logic_symmetry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig


class LogicSymCfg(PluginConfig):
    response_channel: str
    design_by: List[str] = Field(default_factory=lambda: ["genotype"])
    batch_col: str = "batch"
    treatment_map: Dict[str,str]
    treatment_case_sensitive: bool = True
    aggregation: Dict[str, Any] = Field(default_factory=dict)
    encodings: Dict[str, Any]   = Field(default_factory=dict)
    ideals_overlay: Dict[str, Any] = Field(default_factory=dict)
    visuals: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    prep: Optional[Dict[str, Any]] = None
    fig: Dict[str, Any] = Field(default_factory=dict)
    filename: Optional[str] = None

class LogicSymmetryPlot(Plugin):
    key = "logic_symmetry"
    category = "plot"
    ConfigModel = LogicSymCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str,str]:
        # tidy+map helps ensure design_by & batch present; for leniency accept tidy.v1 in earlier range
        return {"df": "tidy+map.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str,str]:
        return {"files": "none"}

    def run(self, ctx, inputs, cfg: LogicSymCfg):
        from reader.lib.logic_symmetry import plot_logic_symmetry
        df: pd.DataFrame = inputs["df"]
        plot_logic_symmetry(
            df=df, blanks=df.iloc[0:0], output_dir=ctx.plots_dir,
            response_channel=cfg.response_channel,
            design_by=cfg.design_by, batch_col=cfg.batch_col, treatment_map=cfg.treatment_map,
            treatment_case_sensitive=cfg.treatment_case_sensitive,
            aggregation=cfg.aggregation, encodings=cfg.encodings, ideals_overlay=cfg.ideals_overlay,
            visuals=cfg.visuals, output=cfg.output, prep=cfg.prep, fig_kwargs=cfg.fig, filename=cfg.filename,
            palette_book=ctx.palette_book,
        )
        return {"files": None}
