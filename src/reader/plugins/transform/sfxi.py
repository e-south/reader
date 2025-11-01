"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/sfxi.py

SFXI: setpoint_fidelity_x_intensity → vec8

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig


class SFXICfg(PluginConfig):
    response: Dict[str, str]            # {"logic_channel":..., "intensity_channel":...}
    design_by: List[str] = Field(default_factory=lambda: ["genotype"])
    batch_col: str = "batch"
    time_mode: str = "nearest"          # nearest|last_before|first_after|exact
    target_time_h: Optional[float] = None
    time_tolerance_h: float = 0.5
    treatment_map: Dict[str, str]
    reference: Dict[str, str | None] = Field(default_factory=lambda: {"genotype": None, "scope": "batch", "stat": "mean"})
    treatment_case_sensitive: bool = True
    require_all_corners_per_design: bool = True
    eps_ratio: float = 1e-9
    eps_range: float = 1e-12
    eps_ref: float = 1e-9
    eps_abs: float = 0.0


class SFXITransform(Plugin):
    key = "sfxi"
    category = "transform"
    ConfigModel = SFXICfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy+map.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"vec8": "sfxi.vec8.v1"}

    def run(self, ctx, inputs, cfg: SFXICfg):
        # Use the new lib location (no legacy processors/)
        from reader.lib.sfxi.math import compute_vec8
        from reader.lib.sfxi.selection import cornerize_and_aggregate

        df: pd.DataFrame = inputs["df"].copy()

        # logic channel
        sel_logic = cornerize_and_aggregate(
            df,
            design_by=cfg.design_by,
            batch_col=cfg.batch_col,
            treatment_map=cfg.treatment_map,
            case_sensitive=cfg.treatment_case_sensitive,
            time_column="time",
            channel=cfg.response["logic_channel"],
            target_time_h=cfg.target_time_h,
            time_mode=cfg.time_mode,
            time_tolerance_h=cfg.time_tolerance_h,
            time_per_batch=True,
            on_missing_time="error",
            require_all_corners_per_design=cfg.require_all_corners_per_design,
        )
        # intensity channel
        sel_int = cornerize_and_aggregate(
            df,
            design_by=cfg.design_by,
            batch_col=cfg.batch_col,
            treatment_map=cfg.treatment_map,
            case_sensitive=cfg.treatment_case_sensitive,
            time_column="time",
            channel=cfg.response["intensity_channel"],
            target_time_h=cfg.target_time_h,
            time_mode=cfg.time_mode,
            time_tolerance_h=cfg.time_tolerance_h,
            time_per_batch=True,
            on_missing_time="error",
            require_all_corners_per_design=cfg.require_all_corners_per_design,
        )

        # compute vec8
        vec8 = compute_vec8(
            points_logic=sel_logic.points,
            points_intensity=sel_int.points,
            per_corner_intensity=sel_int.per_corner,
            design_by=cfg.design_by,
            batch_col=cfg.batch_col,
            reference_genotype=(cfg.reference.get("genotype") if cfg.reference else None),
            reference_scope=(cfg.reference.get("scope") if cfg.reference else "batch"),
            reference_stat=(cfg.reference.get("stat") if cfg.reference else "mean"),
            eps_ratio=cfg.eps_ratio, eps_range=cfg.eps_range,
            eps_ref=cfg.eps_ref, eps_abs=cfg.eps_abs,
        )

        # normalize columns to contract
        if cfg.design_by and cfg.design_by[0] != "genotype" and cfg.design_by[0] in vec8.columns:
            vec8 = vec8.rename(columns={cfg.design_by[0]: "genotype"})
        if "sequence" not in vec8.columns:
            vec8["sequence"] = pd.NA

        cols = [
            "genotype","sequence","r_logic",
            "v00","v10","v01","v11",
            "y00_star","y10_star","y01_star","y11_star",
            "flat_logic",
        ]
        vec8 = vec8[[c for c in cols if c in vec8.columns]].copy()

        return {"vec8": vec8}
