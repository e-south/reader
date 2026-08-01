from __future__ import annotations

import numpy as np
import pandas as pd

from reader_workbench.domains.logic.four_state_vector.builder import (
    FourStateVectorBuildResult,
    build_four_state_vector_from_tidy,
)
from reader_workbench.domains.logic.four_state_vector.treatment_semantics import (
    bind_four_state_vector_treatment_semantics,
)


def build_four_state_vector_plugin_result(*, ctx, df: pd.DataFrame, cfg) -> FourStateVectorBuildResult:
    if ctx.experiment is None:
        raise ValueError("four_state_vector requires experiment semantics in the run context")
    state_space = ctx.experiment.annotations.resolve_ordered_state_space(ref=cfg.state_map_ref)
    semantics = bind_four_state_vector_treatment_semantics(
        state_ids=state_space.state_ids,
        source_column=state_space.column,
        source_values=state_space.source_values,
        case_sensitive=state_space.case_sensitive,
        treatment_column=cfg.treatment_column,
    )
    run_cfg = semantics.inject(cfg.model_dump(exclude={"state_map_ref"}))
    return build_four_state_vector_from_tidy(df.copy(), run_cfg)


def log_four_state_vector_plugin_result(*, ctx, result: FourStateVectorBuildResult) -> None:
    vector = result.vector
    four_state_vector_cfg = result.cfg
    sel_logic = result.sel_logic
    sel_int = result.sel_int
    label_col = four_state_vector_cfg.design_by[0] if four_state_vector_cfg.design_by else "design_id"
    idx_cols = [col for col in four_state_vector_cfg.design_by if col]

    if sel_logic.time_warning:
        ctx.logger.warning("four_state_vector: %s", sel_logic.time_warning)

    try:
        chosen = sel_logic.chosen_time
        flats = int(vector["flat_logic"].sum()) if "flat_logic" in vector.columns else 0
        r_stats = vector["r_logic"].describe() if "r_logic" in vector.columns else None
        ref_info = result.log.get("reference", {})

        if flats:
            fraction = float(result.log.get("flat_logic_fraction", 0.0))
            samples = [str(value) for value in result.log.get("flat_logic_sample_design_ids", ())]
            sample_note = f" Sample design_ids: {', '.join(samples)}." if samples else ""
            ctx.logger.warning(
                "four_state_vector: flat logic detected for %d/%d designs (%.1f%%).%s",
                flats,
                len(vector),
                fraction * 100.0,
                sample_note,
            )

        ctx.logger.info(
            "four_state_vector • inputs: [accent]logic[/accent]=%s → v00..v11  |  [accent]intensity[/accent]=%s → y*00..y*11",
            four_state_vector_cfg.response.logic_channel,
            four_state_vector_cfg.response.intensity_channel,
        )
        ctx.logger.info(
            "four_state_vector • transform semantics:"
            "  v: log2(LOGIC) → per-row min-max → [0,1] (persisted NOT log)."
            "  y*: log2( ((INTENSITY + eps_abs)/max(A + alpha, eps_ref)) + delta ) (persisted in log2)."
        )
        ctx.logger.info(
            "four_state_vector • knobs: eps_ratio=%.1e eps_range=%.1e eps_ref=%.1e eps_abs=%.1e  |  alpha=%.3g delta=%.3g",
            float(four_state_vector_cfg.eps_ratio),
            float(four_state_vector_cfg.eps_range),
            float(four_state_vector_cfg.eps_ref),
            float(four_state_vector_cfg.eps_abs),
            float(four_state_vector_cfg.ref_add_alpha),
            float(four_state_vector_cfg.log2_offset_delta),
        )
        ctx.logger.info(
            "four_state_vector • r_logic definition: per design dynamic range on LOGIC (linear, ε-guarded): max(L_i)/min(L_i)"
        )
        ctx.logger.info(
            "four_state_vector • design_by=%s\n"
            "   time: mode=%s target=%.3g tol=%.3g • chosen=%s\n"
            "   reference: requested=%r → resolved=%r • sequence=%r • observation_stat=%s\n"
            "   rows: per_corner_logic=%d per_corner_intensity=%d vector=%d (flat=%d)\n"
            "   r_logic: median=%.3g iqr=[%.3g, %.3g]",
            ", ".join(four_state_vector_cfg.design_by),
            four_state_vector_cfg.time_mode,
            float(four_state_vector_cfg.target_time_h or np.nan),
            float(four_state_vector_cfg.time_tolerance_h),
            (float(chosen) if chosen is not None else None),
            ref_info.get("design_id"),
            ref_info.get("design_id_resolved"),
            ref_info.get("sequence"),
            ref_info.get("observation_stat"),
            int(len(sel_logic.per_corner)),
            int(len(sel_int.per_corner)),
            int(len(vector)),
            int(flats),
            (float(r_stats["50%"]) if r_stats is not None else float("nan")),
            (float(r_stats["25%"]) if r_stats is not None else float("nan")),
            (float(r_stats["75%"]) if r_stats is not None else float("nan")),
        )

        _log_four_state_vector_observation_summary(
            ctx=ctx,
            design_by=four_state_vector_cfg.design_by,
            idx_cols=idx_cols,
            sel_logic=sel_logic,
            sel_int=sel_int,
        )
        _log_four_state_vector_vector_preview(
            ctx=ctx,
            vector=vector,
            idx_cols=idx_cols,
            label_col=label_col,
            logic_channel=four_state_vector_cfg.response.logic_channel,
            intensity_channel=four_state_vector_cfg.response.intensity_channel,
            sel_logic=sel_logic,
        )
    except Exception:
        pass


def _log_four_state_vector_observation_summary(
    *, ctx, design_by: list[str], idx_cols: list[str], sel_logic, sel_int
) -> None:
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
            logic_observations = [
                int(row.get("n00_L", 0) or 0),
                int(row.get("n10_L", 0) or 0),
                int(row.get("n01_L", 0) or 0),
                int(row.get("n11_L", 0) or 0),
            ]
            intensity_observations = [
                int(row.get("n00_I", 0) or 0),
                int(row.get("n10_I", 0) or 0),
                int(row.get("n01_I", 0) or 0),
                int(row.get("n11_I", 0) or 0),
            ]
            ctx.logger.info(
                "four_state_vector • %s: observations (logic)=%s  (intensity)=%s",
                key,
                logic_observations,
                intensity_observations,
            )
    except Exception:
        pass


def _log_four_state_vector_vector_preview(
    *,
    ctx,
    vector: pd.DataFrame,
    idx_cols: list[str],
    label_col: str,
    logic_channel: str,
    intensity_channel: str,
    sel_logic,
) -> None:
    try:
        observation_count_map: dict[tuple, tuple[int, ...]] = {}
        if not sel_logic.points.empty:
            counts = sel_logic.points.set_index(idx_cols)[["n00", "n10", "n01", "n11"]]
            for key, values in counts.iterrows():
                normalized_key = key if isinstance(key, tuple) else (key,)
                observation_count_map[normalized_key] = tuple(int(item) for item in values.to_list())

        sort_cols = [col for col in [label_col] if col in vector.columns]
        lines: list[str] = []
        for _, row in vector.sort_values(sort_cols).iterrows():
            key = " | ".join(f"{col}={row[col]}" for col in sort_cols)
            v_txt = [f"{float(row[col]):.3f}" for col in ("v00", "v10", "v01", "v11")]
            y_txt = [f"{float(row[col]):.3f}" for col in ("y00_star", "y10_star", "y01_star", "y11_star")]
            observation_key = tuple(row[col] for col in idx_cols if col in vector.columns)
            count_text = ""
            if observation_key in observation_count_map:
                count_text = f"  observation_n={list(observation_count_map[observation_key])}"
            lines.append(
                f"   • {key}: v={v_txt} | y*={y_txt} | r_logic={float(row.get('r_logic', np.nan)):.3g} "
                f"(max/min={float(row.get('r_logic_max', np.nan)):.3g}/{float(row.get('r_logic_min', np.nan)):.3g}; "
                f"corners {row.get('r_logic_corner_max', '?')}/{row.get('r_logic_corner_min', '?')}; "
                f"span_log2={float(row.get('logic_span_log2', np.nan)):.3g}){count_text}"
            )
        if lines:
            more = "" if len(lines) <= 12 else f"\n   … (+{len(lines) - 12} more)"
            ctx.logger.info(
                "four_state_vector • vector per design  [muted](v from log2(%s) min-max; y* is log2 of anchor-normalized %s)[/muted]\n%s%s",
                logic_channel,
                intensity_channel,
                "\n".join(lines[:12]),
                more,
            )
    except Exception:
        pass
