"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/outlier_filter.py

Simple z-score filter per (channel, time).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from reader.core.registry import Plugin, PluginConfig


class OutlierCfg(PluginConfig):
    enable: bool = False
    z_thresh: float = 4.0


class OutlierFilter(Plugin):
    key = "outlier_filter"
    category = "transform"
    ConfigModel = OutlierCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    def run(self, ctx, inputs, cfg: OutlierCfg):
        if not cfg.enable:
            return {"df": inputs["df"].copy()}

        df = inputs["df"].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        def _f(g: pd.DataFrame) -> pd.DataFrame:
            s = g["value"].dropna()
            if s.size <= 1:
                return g
            mu = float(s.mean())
            sd = float(s.std(ddof=1)) if s.size > 1 else 0.0
            if not np.isfinite(sd) or sd <= 0:
                return g
            z = (g["value"] - mu) / sd
            return g.loc[z.abs() <= float(cfg.z_thresh)]

        out = df.groupby(["channel", "time"], group_keys=False).apply(_f)
        return {"df": out}
