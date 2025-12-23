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

from reader.core.registry import Plugin, PluginConfig
from reader.lib.logic_symmetry import plot_logic_symmetry


class LogicSymCfg(PluginConfig):
    response_channel: str
    design_by: list[str] = Field(default_factory=lambda: ["genotype"])
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
        return {"files": "none", "table": "logic_symmetry.v1"}

    def run(self, ctx, inputs, cfg: LogicSymCfg):
        df: pd.DataFrame = inputs["df"]
        result = plot_logic_symmetry(
            df=df,
            blanks=df.iloc[0:0],
            output_dir=ctx.plots_dir,
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
        table = result.table
        try:
            gcols = [c for c in (cfg.design_by + [cfg.batch_col]) if c in table.columns]
            n_groups = table[gcols].drop_duplicates().shape[0] if gcols else len(table)
            formats = [str(x).lower() for x in (cfg.output or {}).get("format", ["pdf"])]
            base = cfg.filename or "logic_symmetry"
            ctx.logger.info(
                "logic_symmetry • wrote plot(s)=[%s] • base=%s • groups=%d • rows=%d • response=%s • design_by=%s • batch_col=%s",
                ", ".join(formats),
                base,
                int(n_groups),
                int(len(table)),
                cfg.response_channel,
                ", ".join(cfg.design_by),
                cfg.batch_col,
            )
        except Exception:
            pass
        return {"files": None, "table": table}
