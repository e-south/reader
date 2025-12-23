"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/blank.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from reader.core.registry import Plugin, PluginConfig


class BlankCfg(PluginConfig):
    method: str = "disregard"  # disregard | subtract
    capture_blanks: bool = True


def _detect_blanks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = False
    for col in ["is_blank", "blank", "isBlank"]:
        if col in out.columns:
            try:
                mask = mask | out[col].astype(bool)
            except Exception:
                mask = mask | out[col].astype(str).str.lower().isin({"true", "1", "t", "yes"})
    for col in ["treatment", "genotype", "sample_type"]:
        if col in out.columns:
            mask = mask | out[col].astype(str).str.contains("blank", case=False, na=False)
    return out[mask].copy() if isinstance(mask, pd.Series) else out.iloc[0:0].copy()


class BlankCorrection(Plugin):
    key = "blank_correction"
    category = "transform"
    ConfigModel = BlankCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1", "blanks": "tidy.v1"}

    def run(self, ctx, inputs, cfg: BlankCfg):
        df: pd.DataFrame = inputs["df"].copy()
        blanks = _detect_blanks(df)
        if cfg.method == "disregard":
            return {"df": df, "blanks": blanks}
        if cfg.method == "subtract":
            if blanks.empty:
                return {"df": df, "blanks": blanks}
            corr = (
                blanks.assign(value=pd.to_numeric(blanks["value"], errors="coerce"))
                .groupby("channel")["value"]
                .median()
                .rename("__blank__")
            )
            out = df.copy()
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
            out = out.join(corr, on="channel")
            out["value"] = out["value"] - out["__blank__"].fillna(0.0)
            out = out.drop(columns="__blank__")
            return {"df": out, "blanks": blanks}
        raise ValueError(f"unknown method {cfg.method}")
