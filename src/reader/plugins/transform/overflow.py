"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/overflow.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from reader.core.registry import Plugin, PluginConfig


class OverflowCfg(PluginConfig):
    action: str = "max"           # max|drop|nan|none
    clip_quantile: float = 0.999

class OverflowHandling(Plugin):
    key = "overflow_handling"
    category = "transform"
    ConfigModel = OverflowCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str,str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str,str]:
        return {"df": "tidy.v1"}

    def run(self, ctx, inputs, cfg: OverflowCfg):
        df = inputs["df"].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        act = cfg.action.lower()
        if act == "none":
            return {"df": df}
        if act == "drop":
            return {"df": df.dropna(subset=["value"])}
        if act == "nan":
            return {"df": df}
        if act == "max":
            thr = df.groupby("channel")["value"].quantile(cfg.clip_quantile).rename("__thr__")
            out = df.join(thr, on="channel")
            out["value"] = np.minimum(out["value"], out["__thr__"])
            return {"df": out.drop(columns="__thr__")}
        raise ValueError(f"unknown overflow action {cfg.action}")
