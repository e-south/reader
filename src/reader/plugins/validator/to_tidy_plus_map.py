"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/validator/to_tidy_plus_map.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Mapping

import pandas as pd
from pydantic import Field

from reader.core.errors import ExecutionError
from reader.core.registry import Plugin, PluginConfig


class PromoteCfg(PluginConfig):
    require_columns: List[str] = Field(default_factory=lambda: ["treatment","genotype","batch"])
    require_non_null: bool = True  # be strict when promoting

class PromoteToTidyPlusMap(Plugin):
    key = "to_tidy_plus_map"
    category = "validator"
    ConfigModel = PromoteCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str,str]:
        # Accept a tidy table with extra metadata columns present
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str,str]:
        # Emit the strict tidy+map contract
        return {"df": "tidy+map.v1"}

    def run(self, ctx, inputs, cfg: PromoteCfg):
        df: pd.DataFrame = inputs["df"].copy()
        missing = [c for c in cfg.require_columns if c not in df.columns]
        if missing:
            raise ExecutionError(f"Cannot promote to tidy+map.v1; missing columns: {missing}")
        if cfg.require_non_null:
            bad = {c: int(df[c].isna().sum()) for c in cfg.require_columns if df[c].isna().any()}
            if bad:
                raise ExecutionError(f"Cannot promote; required columns contain NaN: {bad}")
        # dtype normalization for 'batch' (if present)
        if "batch" in df.columns:
            df["batch"] = pd.to_numeric(df["batch"], errors="raise").astype("Int64")
        return {"df": df}
