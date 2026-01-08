"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/crosstalk_pairs.py

Crosstalk pairing transform over fold_change.v1 tables.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig
from reader.lib.crosstalk import compute_crosstalk_pairs


def _pick_alias(df: pd.DataFrame, base: str | None) -> str | None:
    if not base:
        return None
    alias = f"{base}_alias"
    if base in df.columns:
        return base
    if alias in df.columns:
        return alias
    return None


class CrosstalkPairsCfg(PluginConfig):
    value_column: str = "log2FC"
    value_scale: Literal["log2", "linear"] = Field(..., description="Scale of value_column.")

    target: str | None = None
    time_mode: Literal["single", "exact", "nearest", "latest", "all"] = Field(
        ..., description="How to select time(s) from the fold_change table."
    )
    time_policy: Literal["per_time", "all"] = Field(
        "per_time", description="How to handle multiple times after selection."
    )
    time: float | None = None
    times: list[float] | None = None
    time_tolerance: float = 0.51
    time_column: str = "time"

    mapping_mode: Literal["explicit", "column", "top1"] = Field(
        ..., description="How to map design_id -> cognate treatment."
    )
    design_column: str = "design_id"
    treatment_column: str = "treatment"
    design_treatment_column: str | None = None
    design_treatment_map: dict[str, str] = Field(default_factory=dict)
    design_treatment_overrides: dict[str, str] = Field(default_factory=dict)
    top1_tie_policy: Literal["error", "alphabetical"] = "error"
    top1_tie_tolerance: float = 0.0

    agg: Literal["median", "mean"] = "median"

    require_self_treatment: bool = True
    require_self_is_top1: bool = False

    min_self: float | None = None
    max_cross: float | None = None
    max_other: float | None = None
    min_self_minus_best_other: float | None = None
    min_self_ratio_best_other: float | None = None
    min_selectivity_delta: float | None = None
    min_selectivity_ratio: float | None = None

    only_passing: bool = True
    top_n: int = 10


class CrosstalkPairs(Plugin):
    """Compute crosstalk-safe design pairings from a fold_change.v1 table."""

    key = "crosstalk_pairs"
    category = "transform"
    ConfigModel = CrosstalkPairsCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"table": "fold_change.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"table": "crosstalk_pairs.v1"}

    def run(self, ctx, inputs: dict[str, Any], cfg: CrosstalkPairsCfg) -> dict[str, Any]:
        df = inputs["table"].copy()

        dcol = _pick_alias(df, cfg.design_column)
        tcol = _pick_alias(df, cfg.treatment_column)
        if dcol is None:
            raise ValueError(f"crosstalk_pairs: design column '{cfg.design_column}' (or its alias) is missing")
        if tcol is None:
            raise ValueError(f"crosstalk_pairs: treatment column '{cfg.treatment_column}' (or its alias) is missing")

        map_col = _pick_alias(df, cfg.design_treatment_column) if cfg.design_treatment_column else None

        result = compute_crosstalk_pairs(
            df,
            design_col=dcol,
            treatment_col=tcol,
            value_col=cfg.value_column,
            value_scale=cfg.value_scale,
            target=cfg.target,
            time_mode=cfg.time_mode,
            time_policy=cfg.time_policy,
            time=cfg.time,
            times=cfg.times,
            time_column=cfg.time_column,
            time_tolerance=cfg.time_tolerance,
            mapping_mode=cfg.mapping_mode,
            design_treatment_column=map_col,
            design_treatment_map=cfg.design_treatment_map,
            design_treatment_overrides=cfg.design_treatment_overrides,
            top1_tie_policy=cfg.top1_tie_policy,
            top1_tie_tolerance=cfg.top1_tie_tolerance,
            require_self_treatment=cfg.require_self_treatment,
            require_self_is_top1=cfg.require_self_is_top1,
            agg=cfg.agg,
            min_self=cfg.min_self,
            max_cross=cfg.max_cross,
            max_other=cfg.max_other,
            min_self_minus_best_other=cfg.min_self_minus_best_other,
            min_self_ratio_best_other=cfg.min_self_ratio_best_other,
            min_selectivity_delta=cfg.min_selectivity_delta,
            min_selectivity_ratio=cfg.min_selectivity_ratio,
            only_passing=cfg.only_passing,
            logger=ctx.logger,
        )

        pairs = result.pairs
        designs = result.designs

        try:
            total_designs = int(designs["design_id"].nunique()) if "design_id" in designs.columns else int(0)
            time_count = len(result.times_used)
            total_pairs = total_designs * (total_designs - 1) // 2 * max(1, time_count)
            passing = int(pairs.shape[0])
            ctx.logger.info(
                "crosstalk_pairs - designs=%d - times=%d - pairs=%d - passing=%d - metric=%s",
                total_designs,
                time_count,
                total_pairs,
                passing,
                cfg.value_column,
            )
        except Exception:
            pass

        try:
            if not pairs.empty:
                preview = pairs.copy()
                preview["pair_score"] = pd.to_numeric(preview["pair_score"], errors="coerce")
                preview = preview[preview["pair_score"].notna()].sort_values("pair_score", ascending=False)
                if not preview.empty:
                    n_show = max(1, int(cfg.top_n))
                    ctx.logger.info("crosstalk_pairs - top %d pairs by pair_score:", n_show)
                    for _, row in preview.head(n_show).iterrows():
                        ctx.logger.info(
                            "   - t=%s | %s <-> %s | self=%s/%s | cross=%s/%s | score=%.3g",
                            _fmt(row.get("time")),
                            row.get("design_a"),
                            row.get("design_b"),
                            _fmt(row.get("a_self_value")),
                            _fmt(row.get("b_self_value")),
                            _fmt(row.get("a_cross_to_b")),
                            _fmt(row.get("b_cross_to_a")),
                            float(row.get("pair_score")) if pd.notna(row.get("pair_score")) else float("nan"),
                        )
        except Exception:
            pass

        return {"table": pairs}


def _fmt(val: object) -> str:
    try:
        f = float(val)
        if pd.isna(f):
            return "nan"
        return f"{f:.3g}"
    except Exception:
        return str(val)
