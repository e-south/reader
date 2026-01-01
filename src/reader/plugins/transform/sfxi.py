"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/sfxi.py

SFXI: setpoint_fidelity_x_intensity → vec8 (objective scoring is external)

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig
from reader.domain.sfxi.math import compute_vec8
from reader.domain.sfxi.reference import resolve_reference_design_id
from reader.domain.sfxi.selection import cornerize_and_aggregate


class SFXICfg(PluginConfig):
    response: dict[str, str]  # {"logic_channel":..., "intensity_channel":...}
    design_by: list[str] = Field(default_factory=lambda: ["design_id"])
    time_mode: str = "nearest"  # nearest|last_before|first_after|exact
    target_time_h: float | None = None
    time_tolerance_h: float = 0.5
    treatment_map: dict[str, str]
    reference: dict[str, str | None] = Field(default_factory=lambda: {"design_id": None, "stat": "mean"})
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
        return {"df": "tidy+map.v2"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"vec8": "sfxi.vec8.v2"}

    def run(self, ctx, inputs, cfg: SFXICfg):
        df: pd.DataFrame = inputs["df"].copy()
        label_col = cfg.design_by[0] if cfg.design_by else "design_id"
        idx_cols = [c for c in cfg.design_by if c]
        ref_cfg = cfg.reference or {}
        if "genotype" in ref_cfg:
            raise ValueError("sfxi.reference.genotype is deprecated; use reference.design_id")
        unknown_ref = sorted(set(ref_cfg.keys()) - {"design_id", "stat"})
        if unknown_ref:
            raise ValueError(f"sfxi.reference has unknown keys: {unknown_ref} (allowed: design_id, stat)")

        # ---------- selection (logic channel) ----------
        sel_logic = cornerize_and_aggregate(
            df,
            design_by=cfg.design_by,
            treatment_map=cfg.treatment_map,
            case_sensitive=cfg.treatment_case_sensitive,
            time_column="time",
            channel=cfg.response["logic_channel"],
            target_time_h=cfg.target_time_h,
            time_mode=cfg.time_mode,
            time_tolerance_h=cfg.time_tolerance_h,
            on_missing_time="error",
            require_all_corners_per_design=cfg.require_all_corners_per_design,
        )

        # ---------- selection (intensity channel) ----------
        sel_int = cornerize_and_aggregate(
            df,
            design_by=cfg.design_by,
            treatment_map=cfg.treatment_map,
            case_sensitive=cfg.treatment_case_sensitive,
            time_column="time",
            channel=cfg.response["intensity_channel"],
            target_time_h=cfg.target_time_h,
            time_mode=cfg.time_mode,
            time_tolerance_h=cfg.time_tolerance_h,
            on_missing_time="error",
            require_all_corners_per_design=cfg.require_all_corners_per_design,
        )

        # ---- warn if chosen time differs from target beyond tolerance ----
        if sel_logic.time_warning:
            with contextlib.suppress(Exception):
                ctx.logger.warning("sfxi • %s", sel_logic.time_warning)

        # ---------- assert same chosen times across channels ----------
        ta, tb = float(sel_logic.chosen_time), float(sel_int.chosen_time)
        if not np.isclose(ta, tb, rtol=0, atol=1e-9):
            raise ValueError(f"sfxi: logic and intensity selected different times: {ta} vs {tb}")

        # ---------- resolve & assert reference design_id (to RAW label) ----------
        provided_ref = ref_cfg.get("design_id")
        ref_design = resolve_reference_design_id(inputs["df"], design_by=cfg.design_by, ref_label=provided_ref)
        if ref_design is not None:
            ref_rows = sel_int.per_corner[sel_int.per_corner[label_col].astype(str) == str(ref_design)]
            if ref_rows.empty:
                raise ValueError(
                    f"sfxi: reference design_id {ref_design!r} (resolved from {provided_ref!r}) is not present in the "
                    "INTENSITY channel at the chosen time(s); cannot anchor absolute intensity. "
                    "Ensure a reference strain is included or adjust 'reference.design_id'."
                )

        # ---------- compute vec8 ----------
        vec8 = compute_vec8(
            points_logic=sel_logic.points,
            points_intensity=sel_int.points,
            per_corner_intensity=sel_int.per_corner,
            design_by=cfg.design_by,
            reference_design_id=ref_design,
            reference_stat=(cfg.reference.get("stat") if cfg.reference else "mean"),
            eps_ratio=cfg.eps_ratio,
            eps_range=cfg.eps_range,
            eps_ref=cfg.eps_ref,
            eps_abs=cfg.eps_abs,
            ref_add_alpha=cfg.ref_add_alpha,
            log2_offset_delta=cfg.log2_offset_delta,
        )

        # ---- attach selected metadata columns (e.g., sequence, id) if present ----
        base = inputs["df"]
        for col in cfg.carry_metadata or []:
            if col in base.columns and col not in vec8.columns:
                meta = base[idx_cols + [col]].dropna(subset=[col]).drop_duplicates(subset=idx_cols, keep="first")
                vec8 = vec8.merge(meta, on=idx_cols, how="left", validate="m:1")

        # rename primary label to 'design_id' for output consistency
        if label_col != "design_id" and label_col in vec8.columns:
            vec8 = vec8.rename(columns={label_col: "design_id"})

        # optionally drop reference rows from the output (default: True)
        if cfg.exclude_reference_from_output and ref_design is not None and "design_id" in vec8.columns:
            vec8 = vec8[vec8["design_id"].astype(str) != str(ref_design)].copy()

        if "flat_logic" in vec8.columns:
            flat_count = int(vec8["flat_logic"].sum())
            if flat_count:
                ctx.logger.warning("sfxi • flat logic detected in %d design row(s); v set to 0.25.", flat_count)

        # standard column preference (keep extra metadata too)
        cols = [
            "design_id",
            "sequence",
            "id",
            "r_logic",
            "v00",
            "v10",
            "v01",
            "v11",
            "y00_star",
            "y10_star",
            "y01_star",
            "y11_star",
            "flat_logic",
        ]
        front = [c for c in cols if c in vec8.columns]
        rest = [c for c in vec8.columns if c not in front]
        vec8 = vec8[front + rest].copy()

        # Final type normalization for contract compliance.
        if "flat_logic" in vec8.columns:
            vec8["flat_logic"] = vec8["flat_logic"].astype(bool)

        # ---------- INFO LOGGING ----------
        try:
            chosen = sel_logic.chosen_time  # same as sel_int
            flats = int(vec8["flat_logic"].sum()) if "flat_logic" in vec8.columns else 0
            r_stats = vec8["r_logic"].describe() if "r_logic" in vec8.columns else None

            # 1) transform semantics + one-shot summary
            ctx.logger.info(
                "sfxi • inputs: [accent]logic[/accent]=%s → v00..v11  |  [accent]intensity[/accent]=%s → y*00..y*11",
                cfg.response["logic_channel"],
                cfg.response["intensity_channel"],
            )
            ctx.logger.info(
                "sfxi • transform semantics:"
                "  v: log2(LOGIC) → per-row min-max → [0,1] (persisted NOT log)."
                "  y*: log2( ((INTENSITY + eps_abs)/max(A + alpha, eps_ref)) + delta ) (persisted in log2)."
            )
            ctx.logger.info(
                "sfxi • knobs: eps_ratio=%.1e eps_range=%.1e eps_ref=%.1e eps_abs=%.1e  |  alpha=%.3g delta=%.3g",
                float(cfg.eps_ratio),
                float(cfg.eps_range),
                float(cfg.eps_ref),
                float(cfg.eps_abs),
                float(cfg.ref_add_alpha),
                float(cfg.log2_offset_delta),
            )
            ctx.logger.info(
                "sfxi • r_logic definition: per design dynamic range on LOGIC (linear, ε-guarded): max(L_i)/min(L_i)"
            )
            ctx.logger.info(
                "sfxi • design_by=%s\n"
                "   time: mode=%s target=%.3g tol=%.3g • chosen=%.3g\n"
                "   reference: requested=%r → raw=%r • stat=%s • α=%.3g δ=%.3g\n"
                "   rows: per_corner_logic=%d per_corner_intensity=%d vec8=%d (flat=%d)\n"
                "   r_logic: median=%.3g iqr=[%.3g, %.3g]",
                ", ".join(cfg.design_by),
                cfg.time_mode,
                float(cfg.target_time_h or np.nan),
                float(cfg.time_tolerance_h),
                float(chosen),
                provided_ref,
                ref_design,
                (cfg.reference or {}).get("stat", "mean"),
                float(cfg.ref_add_alpha),
                float(cfg.log2_offset_delta),
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
                # pretty lines
                preview_lines = []
                n_join = n_join.sort_values(idx_cols)
                for _, rr in n_join.iterrows():
                    key = " | ".join(f"{c}={rr[c]}" for c in cfg.design_by if c in rr.index)
                    L = [
                        int(rr.get("n00_L", 0) or 0),
                        int(rr.get("n10_L", 0) or 0),
                        int(rr.get("n01_L", 0) or 0),
                        int(rr.get("n11_L", 0) or 0),
                    ]
                    I_counts = [
                        int(rr.get("n00_I", 0) or 0),
                        int(rr.get("n10_I", 0) or 0),  # noqa
                        int(rr.get("n01_I", 0) or 0),
                        int(rr.get("n11_I", 0) or 0),
                    ]
                    preview_lines.append(f"   • {key}: replicates (logic)={L}  (intensity)={I_counts}")

            except Exception:
                pass

            # pretty vec8 preview per design_id
            try:
                # map replicate counts for convenience (logic only)
                rep_map = {}
                if not nL.empty:
                    for k, vals in (
                        nL.reset_index().set_index(idx_cols)[["n00_L", "n10_L", "n01_L", "n11_L"]].iterrows()
                    ):
                        rep_map[k] = tuple(int(x) for x in vals.to_list())
                gcol = "design_id" if "design_id" in vec8.columns else label_col
                sort_cols = [c for c in [gcol] if c in vec8.columns]
                lines = []
                for _, r in vec8.sort_values(sort_cols).iterrows():
                    key = " | ".join(f"{c}={r[c]}" for c in sort_cols)
                    v = [float(r["v00"]), float(r["v10"]), float(r["v01"]), float(r["v11"])]
                    y = [float(r["y00_star"]), float(r["y10_star"]), float(r["y01_star"]), float(r["y11_star"])]
                    # r_logic details if present
                    rlog = float(r.get("r_logic", np.nan))
                    rmin = r.get("r_logic_min", np.nan)
                    rmax = r.get("r_logic_max", np.nan)
                    span = r.get("logic_span_log2", np.nan)
                    cmin = r.get("r_logic_corner_min", "?")
                    cmax = r.get("r_logic_corner_max", "?")
                    # replicate counts (if available)
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
                        cfg.response["logic_channel"],
                        cfg.response["intensity_channel"],
                        "\n".join(lines[:12]),
                        more,
                    )
            except Exception:
                pass

        except Exception:
            pass

        return {"vec8": vec8}
