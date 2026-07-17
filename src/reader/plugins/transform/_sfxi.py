from __future__ import annotations

import numpy as np
import pandas as pd

from reader.domains.logic.sfxi.run import SFXIBuildResult, build_vec8_from_tidy
from reader.domains.logic.sfxi.treatment_semantics import resolve_sfxi_treatment_semantics


def build_sfxi_plugin_result(*, ctx, df: pd.DataFrame, cfg) -> SFXIBuildResult:
    semantics = resolve_sfxi_treatment_semantics(
        ctx=ctx,
        state_map_ref=cfg.state_map_ref,
        treatment_column=cfg.treatment_column,
    )
    run_cfg = semantics.inject(cfg.model_dump())
    return build_vec8_from_tidy(df.copy(), run_cfg)


def log_sfxi_plugin_result(*, ctx, result: SFXIBuildResult) -> None:
    vec8 = result.vec8
    sfxi_cfg = result.cfg
    sel_logic = result.sel_logic
    sel_int = result.sel_int
    label_col = sfxi_cfg.design_by[0] if sfxi_cfg.design_by else "design_id"
    idx_cols = [col for col in sfxi_cfg.design_by if col]

    if sel_logic.time_warning:
        ctx.logger.warning("sfxi: %s", sel_logic.time_warning)

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

        _log_sfxi_replicate_summary(
            ctx=ctx,
            design_by=sfxi_cfg.design_by,
            idx_cols=idx_cols,
            sel_logic=sel_logic,
            sel_int=sel_int,
        )
        _log_sfxi_vec8_preview(
            ctx=ctx,
            vec8=vec8,
            idx_cols=idx_cols,
            label_col=label_col,
            logic_channel=sfxi_cfg.response.logic_channel,
            intensity_channel=sfxi_cfg.response.intensity_channel,
            sel_logic=sel_logic,
        )
    except Exception:
        pass


def _log_sfxi_replicate_summary(*, ctx, design_by: list[str], idx_cols: list[str], sel_logic, sel_int) -> None:
    try:
        logic_counts = sel_logic.points.set_index(idx_cols)[["n00", "n10", "n01", "n11"]].rename(
            columns={"n00": "n00_L", "n10": "n10_L", "n01": "n01_L", "n11": "n11_L"}
        )
        intensity_counts = sel_int.points.set_index(idx_cols)[["n00", "n10", "n01", "n11"]].rename(
            columns={"n00": "n00_I", "n10": "n10_I", "n01": "n01_I", "n11": "n11_I"}
        )
        joined = logic_counts.join(intensity_counts, how="outer").reset_index().sort_values(idx_cols)
        for _, row in joined.iterrows():
            key = " | ".join(f"{col}={row[col]}" for col in design_by if col in row.index)
            logic_reps = [
                int(row.get("n00_L", 0) or 0),
                int(row.get("n10_L", 0) or 0),
                int(row.get("n01_L", 0) or 0),
                int(row.get("n11_L", 0) or 0),
            ]
            intensity_reps = [
                int(row.get("n00_I", 0) or 0),
                int(row.get("n10_I", 0) or 0),
                int(row.get("n01_I", 0) or 0),
                int(row.get("n11_I", 0) or 0),
            ]
            ctx.logger.info("sfxi • %s: replicates (logic)=%s  (intensity)=%s", key, logic_reps, intensity_reps)
    except Exception:
        pass


def _log_sfxi_vec8_preview(
    *,
    ctx,
    vec8: pd.DataFrame,
    idx_cols: list[str],
    label_col: str,
    logic_channel: str,
    intensity_channel: str,
    sel_logic,
) -> None:
    try:
        rep_map: dict[tuple, tuple[int, ...]] = {}
        if not sel_logic.points.empty:
            counts = sel_logic.points.set_index(idx_cols)[["n00", "n10", "n01", "n11"]]
            for key, values in counts.iterrows():
                normalized_key = key if isinstance(key, tuple) else (key,)
                rep_map[normalized_key] = tuple(int(item) for item in values.to_list())

        sort_cols = [col for col in [label_col] if col in vec8.columns]
        lines: list[str] = []
        for _, row in vec8.sort_values(sort_cols).iterrows():
            key = " | ".join(f"{col}={row[col]}" for col in sort_cols)
            v_txt = [f"{float(row[col]):.3f}" for col in ("v00", "v10", "v01", "v11")]
            y_txt = [f"{float(row[col]):.3f}" for col in ("y00_star", "y10_star", "y01_star", "y11_star")]
            rep_key = tuple(row[col] for col in idx_cols if col in vec8.columns)
            rep_txt = ""
            if rep_key in rep_map:
                rep_txt = f"  n={list(rep_map[rep_key])}"
            lines.append(
                f"   • {key}: v={v_txt} | y*={y_txt} | r_logic={float(row.get('r_logic', np.nan)):.3g} "
                f"(max/min={float(row.get('r_logic_max', np.nan)):.3g}/{float(row.get('r_logic_min', np.nan)):.3g}; "
                f"corners {row.get('r_logic_corner_max', '?')}/{row.get('r_logic_corner_min', '?')}; "
                f"span_log2={float(row.get('logic_span_log2', np.nan)):.3g}){rep_txt}"
            )
        if lines:
            more = "" if len(lines) <= 12 else f"\n   … (+{len(lines) - 12} more)"
            ctx.logger.info(
                "sfxi • vec8 per design  [muted](v from log2(%s) min-max; y* is log2 of anchor-normalized %s)[/muted]\n%s%s",
                logic_channel,
                intensity_channel,
                "\n".join(lines[:12]),
                more,
            )
    except Exception:
        pass
