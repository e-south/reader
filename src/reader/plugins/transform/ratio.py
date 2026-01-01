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

from reader.core.errors import TransformError
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

        # Validate channels exist
        channels = set(df["channel"].astype(str).unique().tolist())
        missing = [c for c in (cfg.numerator, cfg.denominator) if c not in channels]
        if missing:
            preview = ", ".join(sorted(channels)[:8]) + (" …" if len(channels) > 8 else "")
            raise TransformError(f"ratio • {cfg.name}: missing channel(s) {missing}. Available channels: {preview}")

        # Partition numerator/denominator; keep ALL metadata on numerator side
        lhs = df[df["channel"] == cfg.numerator].rename(columns={"value": "__num__"}).copy()
        rhs = df[df["channel"] == cfg.denominator].rename(columns={"value": "__den__"}).copy()

        # Keep only join keys + denominator on RHS to avoid suffix collisions
        rhs = rhs[key + ["__den__"]]

        # Join (lhs may be many-to-one vs rhs on the key)
        merged = pd.merge(lhs, rhs, on=key, how="inner", validate="many_to_one")
        if merged.empty:
            raise TransformError(
                f"ratio • {cfg.name}: no aligned rows after joining on {key}. "
                "Check align_on keys and ensure numerator/denominator rows share the same keys."
            )

        # Numerics + validity filter (no silent drops)
        merged["__num__"] = pd.to_numeric(merged["__num__"], errors="coerce")
        merged["__den__"] = pd.to_numeric(merged["__den__"], errors="coerce")
        bad_num = int(merged["__num__"].isna().sum())
        bad_den = int(merged["__den__"].isna().sum())
        zero_den = int((merged["__den__"] == 0).sum())
        if bad_num or bad_den or zero_den:
            raise TransformError(
                f"ratio • {cfg.name}: invalid values (non-numeric or zero denominator). "
                f"bad_num={bad_num} bad_den={bad_den} zero_den={zero_den}"
            )

        merged["value"] = merged["__num__"] / merged["__den__"]
        merged["channel"] = cfg.name

        # Restore original column set in original order (inherits metadata from lhs)
        derived = merged[df.columns].copy()
        if derived.empty:
            raise TransformError(f"ratio • {cfg.name}: produced zero derived rows. Check align_on keys and input data.")

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
