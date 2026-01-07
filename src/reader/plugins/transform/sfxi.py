"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/sfxi.py

SFXI: setpoint_fidelity_x_intensity → vec8

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig
from reader.lib.sfxi.run import build_vec8_from_tidy


class SFXICfg(PluginConfig):
    response: dict[str, str]  # {"logic_channel":..., "intensity_channel":...}
    design_by: list[str] = Field(default_factory=lambda: ["design_id"])
    time_column: str = "time"
    time_mode: str = "nearest"  # nearest|last_before|first_after|exact
    target_time_h: float | None = None
    time_tolerance_h: float = 0.5
    treatment_map: dict[str, str]
    reference: dict[str, str | None] = Field(
        default_factory=lambda: {"design_id": None, "stat": "mean"}
    )
    treatment_case_sensitive: bool = True
    require_all_corners_per_design: bool = True
    eps_ratio: float = 1e-9
    eps_range: float = 1e-12
    eps_ref: float = 1e-9
    eps_abs: float = 0.0
    ref_add_alpha: float = 0.0
    log2_offset_delta: float = 0.0
    exclude_reference_from_output: bool = True
    carry_metadata: list[str] = Field(default_factory=lambda: ["sequence", "id"])


class SFXITransform(Plugin):
    key = "sfxi"
    category = "transform"
    ConfigModel = SFXICfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy+map.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"vec8": "sfxi.vec8.v2"}

    def run(self, ctx, inputs, cfg: SFXICfg):
        df: pd.DataFrame = inputs["df"].copy()
        result = build_vec8_from_tidy(df, cfg)
        vec8 = result.vec8
        sfxi_cfg = result.cfg
        sel_logic = result.sel_logic
        sel_int = result.sel_int
        label_col = sfxi_cfg.design_by[0] if sfxi_cfg.design_by else "design_id"
        idx_cols = [c for c in sfxi_cfg.design_by if c]

        if sel_logic.time_warning:
            ctx.logger.warning("sfxi: %s", sel_logic.time_warning)

        # ---------- INFO LOGGING ----------
        try:
            chosen = sel_logic.chosen_time
            flats = int(vec8["flat_logic"].sum()) if "flat_logic" in vec8.columns else 0
            r_stats = vec8["r_logic"].describe() if "r_logic" in vec8.columns else None
            ref_info = result.log.get("reference", {})

            ctx.logger.info(
                "sfxi • inputs: [accent]logic[/accent]=%s → v00..v11  |  [accent]intensity[/accent]=%s → y*00..y*11",
                sfxi_cfg.response.logic_channel,
                sfxi_cfg.response.intensity_channel,
            )
            ctx.logger.info(
                "sfxi • transform semantics:"
                "  v: log2(LOGIC) → per-row min-max → [0,1] (persisted NOT log)."
                "  y*: log2( ((INTENSITY + eps_abs)/max(A + alpha, eps_ref)) + delta ) (persisted in log2)."
            )
            ctx.logger.info(
                "sfxi • knobs: eps_ratio=%.1e eps_range=%.1e eps_ref=%.1e eps_abs=%.1e  |  alpha=%.3g delta=%.3g",
                float(sfxi_cfg.eps_ratio),
                float(sfxi_cfg.eps_range),
                float(sfxi_cfg.eps_ref),
                float(sfxi_cfg.eps_abs),
                float(sfxi_cfg.ref_add_alpha),
                float(sfxi_cfg.log2_offset_delta),
            )
            ctx.logger.info(
                "sfxi • r_logic definition: per design dynamic range on LOGIC (linear, ε-guarded): max(L_i)/min(L_i)"
            )
            ctx.logger.info(
                "sfxi • design_by=%s\n"
                "   time: mode=%s target=%.3g tol=%.3g • chosen=%s\n"
                "   reference: requested=%r → resolved=%r • sequence=%r • stat=%s\n"
                "   rows: per_corner_logic=%d per_corner_intensity=%d vec8=%d (flat=%d)\n"
                "   r_logic: median=%.3g iqr=[%.3g, %.3g]",
                ", ".join(sfxi_cfg.design_by),
                sfxi_cfg.time_mode,
                float(sfxi_cfg.target_time_h or np.nan),
                float(sfxi_cfg.time_tolerance_h),
                (float(chosen) if chosen is not None else None),
                ref_info.get("design_id"),
                ref_info.get("design_id_resolved"),
                ref_info.get("sequence"),
                ref_info.get("stat"),
                int(len(sel_logic.per_corner)),
                int(len(sel_int.per_corner)),
                int(len(vec8)),
                int(flats),
                (float(r_stats["50%"]) if r_stats is not None else float("nan")),
                (float(r_stats["25%"]) if r_stats is not None else float("nan")),
                (float(r_stats["75%"]) if r_stats is not None else float("nan")),
            )

            # replicate summary per design (logic & intensity)
            try:
                nL = sel_logic.points.set_index(idx_cols)[["n00", "n10", "n01", "n11"]].rename(
                    columns={"n00": "n00_L", "n10": "n10_L", "n01": "n01_L", "n11": "n11_L"}
                )
                nI = sel_int.points.set_index(idx_cols)[["n00", "n10", "n01", "n11"]].rename(
                    columns={"n00": "n00_I", "n10": "n10_I", "n01": "n01_I", "n11": "n11_I"}
                )
                n_join = nL.join(nI, how="outer").reset_index()
                n_join = n_join.sort_values(idx_cols)
                for _, rr in n_join.iterrows():
                    key = " | ".join(f"{c}={rr[c]}" for c in sfxi_cfg.design_by if c in rr.index)
                    L = [
                        int(rr.get("n00_L", 0) or 0),
                        int(rr.get("n10_L", 0) or 0),
                        int(rr.get("n01_L", 0) or 0),
                        int(rr.get("n11_L", 0) or 0),
                    ]
                    I_counts = [
                        int(rr.get("n00_I", 0) or 0),
                        int(rr.get("n10_I", 0) or 0),
                        int(rr.get("n01_I", 0) or 0),
                        int(rr.get("n11_I", 0) or 0),
                    ]
                    ctx.logger.info("sfxi • %s: replicates (logic)=%s  (intensity)=%s", key, L, I_counts)
            except Exception:
                pass

            # pretty vec8 preview per design
            try:
                rep_map = {}
                if "nL" in locals() and not nL.empty:
                    for k, vals in (
                        nL.reset_index().set_index(idx_cols)[["n00_L", "n10_L", "n01_L", "n11_L"]].iterrows()
                    ):
                        rep_map[k] = tuple(int(x) for x in vals.to_list())
                sort_cols = [c for c in [label_col] if c in vec8.columns]
                lines = []
                for _, r in vec8.sort_values(sort_cols).iterrows():
                    key = " | ".join(f"{c}={r[c]}" for c in sort_cols)
                    v = [float(r["v00"]), float(r["v10"]), float(r["v01"]), float(r["v11"])]
                    y = [float(r["y00_star"]), float(r["y10_star"]), float(r["y01_star"]), float(r["y11_star"])]
                    rlog = float(r.get("r_logic", np.nan))
                    rmin = r.get("r_logic_min", np.nan)
                    rmax = r.get("r_logic_max", np.nan)
                    span = r.get("logic_span_log2", np.nan)
                    cmin = r.get("r_logic_corner_min", "?")
                    cmax = r.get("r_logic_corner_max", "?")
                    rep_key = tuple(r[c] for c in idx_cols if c in vec8.columns)
                    reps = rep_map.get(rep_key)
                    rep_txt = "" if reps is None else f"  n={list(reps)}"
                    v_txt = [f"{x:.3f}" for x in v]
                    y_txt = [f"{x:.3f}" for x in y]
                    lines.append(
                        f"   • {key}: v={v_txt} | y*={y_txt} | r_logic={rlog:.3g} "
                        f"(max/min={float(rmax):.3g}/{float(rmin):.3g}; corners {cmax}/{cmin}; "
                        f"span_log2={float(span):.3g}){rep_txt}"
                    )
                if lines:
                    more = "" if len(lines) <= 12 else f"\n   … (+{len(lines) - 12} more)"
                    ctx.logger.info(
                        "sfxi • vec8 per design  "
                        "[muted](v from log2(%s) min-max; y* is log2 of anchor-normalized %s)[/muted]\n%s%s",
                        sfxi_cfg.response.logic_channel,
                        sfxi_cfg.response.intensity_channel,
                        "\n".join(lines[:12]),
                        more,
                    )
            except Exception:
                pass

        except Exception:
            pass

        return {"vec8": vec8}
