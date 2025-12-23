"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/validator/to_tidy_plus_map.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress

import pandas as pd
from pydantic import Field

from reader.core.errors import ExecutionError
from reader.core.registry import Plugin, PluginConfig


class PromoteCfg(PluginConfig):
    require_columns: list[str] = Field(default_factory=lambda: ["treatment", "genotype", "batch"])
    require_non_null: bool = True  # be strict when promoting
    # Only promote a subset of rows (e.g., samples). If provided, we require the column to exist.
    type_column: str = "type"
    include_types: list[str] = Field(default_factory=list)  # e.g., ["SAMPLE"]
    # Deterministically drop rows with NULL in these columns before assertions.
    drop_where_null_in: list[str] = Field(default_factory=list)


class PromoteToTidyPlusMap(Plugin):
    key = "to_tidy_plus_map"
    category = "validator"
    ConfigModel = PromoteCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        # Accept a tidy table with extra metadata columns present
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        # Emit the strict tidy+map contract
        return {"df": "tidy+map.v1"}

    def run(self, ctx, inputs, cfg: PromoteCfg):
        df: pd.DataFrame = inputs["df"].copy()

        # 0) Optional row filtering by "type" (case-insensitive exact match)
        if cfg.include_types:
            if cfg.type_column not in df.columns:
                raise ExecutionError(
                    f"to_tidy_plus_map: include_types was provided but column {cfg.type_column!r} is missing"
                )
            keep = {str(x).casefold() for x in cfg.include_types}
            before = len(df)
            df = df[df[cfg.type_column].astype(str).str.casefold().isin(keep)].copy()
            with suppress(Exception):
                ctx.logger.info(
                    "to_tidy_plus_map: filtered rows by %r ∈ %s → kept %d/%d",
                    cfg.type_column,
                    sorted(cfg.include_types),
                    len(df),
                    before,
                )
            if df.empty:
                raise ExecutionError(
                    f"to_tidy_plus_map: no rows remain after filtering by {cfg.type_column} ∈ {cfg.include_types}"
                )

        # 0b) Optional: drop rows with NULL in specific columns
        if cfg.drop_where_null_in:
            missing_cols = [c for c in cfg.drop_where_null_in if c not in df.columns]
            if missing_cols:
                raise ExecutionError(f"to_tidy_plus_map: drop_where_null_in refers to missing columns: {missing_cols}")
            before = len(df)
            df = df.dropna(subset=list(cfg.drop_where_null_in)).copy()
            with suppress(Exception):
                ctx.logger.info(
                    "to_tidy_plus_map: dropped %d row(s) with NULL in %s",
                    before - len(df),
                    list(cfg.drop_where_null_in),
                )

        # ---- Make 'batch' optional: synthesize a single-batch when absent or partially missing.
        if "batch" not in df.columns or df["batch"].isna().all():
            df = df.copy()
            df["batch"] = 0
            with suppress(Exception):
                ctx.logger.info("to_tidy_plus_map: 'batch' missing → added constant 0 for all rows")
        elif df["batch"].isna().any():
            df["batch"] = df["batch"].fillna(0)
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
