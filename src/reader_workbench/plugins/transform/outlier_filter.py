"""Simple z-score filter per (channel, time)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reader_workbench.workbench.ports import dataframe_input, dataframe_output
from reader_workbench.workbench.registry import Plugin, PluginConfig


class OutlierCfg(PluginConfig):
    enable: bool = False
    z_thresh: float = 4.0


class OutlierFilter(Plugin):
    ConfigModel = OutlierCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return cls.passthrough_output_ports(
            outputs={"df": dataframe_output("df", "tidy.v1")},
            passthrough={"df": "df"},
            promoted_examples={"df": ("plate_reader.annotated.v1",)},
        )

    def resolve_output_ports(self, *, inputs, outputs, cfg, where):
        del cfg
        return self.inherit_dataframe_output_ports(
            inputs=inputs,
            outputs=outputs,
            passthrough={"df": "df"},
            where=where,
        )

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
