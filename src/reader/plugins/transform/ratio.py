"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/ratio.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig


class RatioCfg(PluginConfig):
    name: str
    numerator: str
    denominator: str
    align_on: list[str] = Field(default_factory=lambda: ["position", "time"])


class RatioTransform(Plugin):
    key = "ratio"
    category = "transform"
    ConfigModel = RatioCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    def run(self, ctx, inputs, cfg: RatioCfg):
        df: pd.DataFrame = inputs["df"].copy()

        # Build alignment key; auto-augment with per-sheet/scope cols if present
        key = [c for c in cfg.align_on if c in df.columns]
        for extra in ("sheet_index", "sheet_name", "source"):
            if extra in df.columns and extra not in key:
                key.append(extra)

        # Partition numerator/denominator; keep ALL metadata on numerator side
        lhs = df[df["channel"] == cfg.numerator].rename(columns={"value": "__num__"}).copy()
        rhs = df[df["channel"] == cfg.denominator].rename(columns={"value": "__den__"}).copy()

        # Keep only join keys + denominator on RHS to avoid suffix collisions
        rhs = rhs[key + ["__den__"]]

        # Join (lhs may be many-to-one vs rhs on the key)
        merged = pd.merge(lhs, rhs, on=key, how="inner", validate="many_to_one")

        # Numerics + validity filter (drop invalids to satisfy tidy.v1: no NaNs)
        merged["__num__"] = pd.to_numeric(merged["__num__"], errors="coerce")
        merged["__den__"] = pd.to_numeric(merged["__den__"], errors="coerce")
        ok = merged["__num__"].notna() & merged["__den__"].notna() & (merged["__den__"] != 0)
        dropped = int((~ok).sum())
        if dropped:
            ctx.logger.warning(
                "[warn]ratio[/warn] • %s: dropped %d row(s) due to missing/zero denominator", cfg.name, dropped
            )

        merged = merged.loc[ok].copy()
        merged["value"] = merged["__num__"] / merged["__den__"]
        merged["channel"] = cfg.name

        # Restore original column set in original order (inherits metadata from lhs)
        derived = merged[df.columns].copy()

        out = pd.concat([df, derived], ignore_index=True)

        with suppress(Exception):
            ctx.logger.info(
                "ratio • [accent]%s[/accent] = %s / %s • +%d row(s) • keys=%s",
                cfg.name,
                cfg.numerator,
                cfg.denominator,
                len(derived),
                key,
            )

        return {"df": out}
