"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/sfxi/run.py

Program entry for SFXI: tidy_data → vec8

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .api import SFXIConfig, load_sfxi_config
from .math import compute_vec8
from .reference import compute_reference_table, resolve_reference_design_id
from .selection import CornerizeResult, cornerize_and_aggregate
from .writer import write_outputs


def _assert_same_times(time_a: float | None, time_b: float | None) -> None:
    if time_a is None and time_b is None:
        return
    if time_a is None or time_b is None:
        raise ValueError("SFXI: logic and intensity selections produced different times.")
    ta = float(time_a)
    tb = float(time_b)
    if not np.isclose(ta, tb, rtol=0, atol=1e-9):
        raise ValueError(f"SFXI: logic and intensity channels selected different times: {ta} vs {tb}")


def _attach_sequence(vec8: pd.DataFrame, tidy_df: pd.DataFrame, *, design_by: list[str]) -> pd.DataFrame:
    # If vec8 already has a usable 'sequence' column, keep it.
    if "sequence" in vec8.columns and vec8["sequence"].notna().any():
        return vec8

    if "sequence" not in tidy_df.columns:
        return vec8

    if "sequence" not in tidy_df.columns:
        return vec8

    idx_cols = [c for c in design_by if c in tidy_df.columns]
    if not idx_cols:
        return vec8

    seq_map = (
        tidy_df[idx_cols + ["sequence"]].dropna(subset=["sequence"]).drop_duplicates(subset=idx_cols, keep="first")
    )

    merged = vec8.merge(seq_map, on=idx_cols, how="left", suffixes=("", "_ref"))
    # Prefer existing 'sequence' if present; otherwise use attached value
    if "sequence_ref" in merged.columns:
        if "sequence" not in merged.columns:
            merged["sequence"] = merged["sequence_ref"]
        else:
            merged["sequence"] = merged["sequence"].fillna(merged["sequence_ref"])
        merged = merged.drop(columns=["sequence_ref"])

    return merged


def _reorder_and_filter(
    vec8: pd.DataFrame,
    *,
    design_by: list[str],
    ref_design_id: str | None,
    exclude_reference: bool,
) -> pd.DataFrame:
    """
    Apply user-requested:
      - drop REF design_id rows
      - reorder columns to:
        design_id, sequence, time_selected_h, reference_design_id, r_logic, v00, v10, v01, v11,
        y00_star, y10_star, y01_star, y11_star, flat_logic (then keep remaining columns)
    """
    label_col = design_by[0]

    df = vec8.copy()

    # Exclude REF (if specified)
    if exclude_reference and ref_design_id is not None and label_col in df.columns:
        df = df[df[label_col].astype(str) != str(ref_design_id)].copy()

    # Ensure 'sequence' exists even if missing upstream
    if "sequence" not in df.columns:
        df["sequence"] = pd.NA

    if "reference_sequence" in df.columns:
        df = df.drop(columns=["reference_sequence"])

    if label_col not in df.columns:
        raise ValueError(f"SFXI: expected label column '{label_col}' in vec8 output.")

    desired = [
        label_col,
        "sequence",
        "time_selected_h",
        "reference_design_id",
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

    # Reorder preferred columns first, then keep remaining diagnostics/metadata.
    front = [c for c in desired if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    out = df[front + rest].copy()

    # Final tidy: drop duplicates just in case and reset index
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def _anchors_payload(anchor_table: pd.DataFrame | None) -> dict[str, float] | None:
    if anchor_table is None or anchor_table.empty:
        return None
    row = anchor_table.iloc[0]
    return {
        "00": float(row.get("b00")),
        "10": float(row.get("b10")),
        "01": float(row.get("b01")),
        "11": float(row.get("b11")),
    }


@dataclass(frozen=True)
class SFXIBuildResult:
    vec8: pd.DataFrame
    log: dict[str, Any]
    cfg: SFXIConfig
    sel_logic: CornerizeResult
    sel_int: CornerizeResult
    ref_design_id: str | None


def build_vec8_from_tidy(tidy_df: pd.DataFrame, xform_cfg: Any) -> SFXIBuildResult:
    """
    Shared SFXI builder for pipeline + notebooks.
    Returns vec8 + log payload + selection diagnostics without writing to disk.
    """
    cfg = load_sfxi_config(xform_cfg)
    if cfg.reference.design_id is None:
        raise ValueError("sfxi.reference.design_id must be provided to anchor intensity.")

    # 1) Cornerize + aggregate (LOGIC channel)
    sel_logic = cornerize_and_aggregate(
        tidy_df,
        design_by=cfg.design_by,
        treatment_map=cfg.treatment_map,
        case_sensitive=cfg.treatment_case_sensitive,
        time_column=cfg.time_column,
        channel=cfg.response.logic_channel,
        target_time_h=cfg.target_time_h,
        time_mode=cfg.time_mode,
        time_tolerance_h=cfg.time_tolerance_h,
        require_all_corners_per_design=cfg.require_all_corners_per_design,
    )

    # 2) Cornerize + aggregate (INTENSITY channel)
    sel_int = cornerize_and_aggregate(
        tidy_df,
        design_by=cfg.design_by,
        treatment_map=cfg.treatment_map,
        case_sensitive=cfg.treatment_case_sensitive,
        time_column=cfg.time_column,
        channel=cfg.response.intensity_channel,
        target_time_h=cfg.target_time_h,
        time_mode=cfg.time_mode,
        time_tolerance_h=cfg.time_tolerance_h,
        require_all_corners_per_design=cfg.require_all_corners_per_design,
    )

    # 3) Safety: same chosen snapshot times
    _assert_same_times(sel_logic.chosen_time, sel_int.chosen_time)

    tol_note = sel_logic.time_warning

    # 4) Compute vec8 (logic from logic channel, y* from intensity channel + anchors)
    ref_raw = resolve_reference_design_id(tidy_df, design_by=cfg.design_by, ref_label=cfg.reference.design_id)
    vec8 = compute_vec8(
        points_logic=sel_logic.points,
        points_intensity=sel_int.points,
        per_corner_intensity=sel_int.per_corner,
        design_by=cfg.design_by,
        reference_design_id=ref_raw,
        reference_stat=cfg.reference.stat,
        eps_ratio=cfg.eps_ratio,
        eps_range=cfg.eps_range,
        eps_ref=cfg.eps_ref,
        eps_abs=cfg.eps_abs,
        ref_add_alpha=cfg.ref_add_alpha,
        log2_offset_delta=cfg.log2_offset_delta,
    )

    # 4a) Attach selected snapshot time
    if sel_logic.chosen_time is not None:
        vec8["time_selected_h"] = float(sel_logic.chosen_time)
        vec8["time_selected_h"] = pd.to_numeric(vec8["time_selected_h"], errors="coerce").astype(float)

    # 4b) Attach 'sequence' if available in the tidy inputs
    vec8 = _attach_sequence(vec8, tidy_df, design_by=cfg.design_by)
    # 4b.5) Persist intensity log2 offset delta for downstream enforcement
    if "intensity_log2_offset_delta" not in vec8.columns:
        vec8["intensity_log2_offset_delta"] = float(cfg.log2_offset_delta)
    # Carry additional metadata columns if requested
    idx_cols = [c for c in cfg.design_by if c in tidy_df.columns]
    for col in cfg.carry_metadata or []:
        if col in tidy_df.columns and col not in vec8.columns:
            meta = tidy_df[idx_cols + [col]].dropna(subset=[col]).drop_duplicates(subset=idx_cols, keep="first")
            vec8 = vec8.merge(meta, on=idx_cols, how="left", validate="m:1")

    # 4c) Reference provenance columns
    if "reference_design_id" not in vec8.columns:
        vec8["reference_design_id"] = ref_raw

    # 4d) Apply requested filtering and column order
    vec8_out = _reorder_and_filter(
        vec8,
        design_by=cfg.design_by,
        ref_design_id=ref_raw,
        exclude_reference=cfg.exclude_reference_from_output,
    )

    # 5) Compose a log payload (unchanged)
    # Compute r_logic distribution for quick provenance metrics
    r_desc = vec8_out["r_logic"].describe().to_dict() if "r_logic" in vec8_out.columns else {}
    flat_count = 0
    flat_fraction = 0.0
    flat_samples: list[str] = []
    if "flat_logic" in vec8_out.columns and len(vec8_out) > 0:
        flat_mask = vec8_out["flat_logic"].astype(bool)
        flat_count = int(flat_mask.sum())
        flat_fraction = float(flat_count) / float(len(vec8_out)) if len(vec8_out) else 0.0
        label_col = cfg.design_by[0] if cfg.design_by else None
        if label_col and label_col in vec8_out.columns and flat_count > 0:
            flat_samples = vec8_out.loc[flat_mask, label_col].astype(str).dropna().drop_duplicates().head(5).tolist()

    log_payload: dict[str, Any] = {
        "name": cfg.name,
        "design_by": cfg.design_by,
        "time_column": cfg.time_column,
        "channels": {
            "logic": cfg.response.logic_channel,
            "intensity": cfg.response.intensity_channel,
        },
        "treatment_map": cfg.treatment_map,
        "treatment_case_sensitive": cfg.treatment_case_sensitive,
        "time": {
            "target_time_h": cfg.target_time_h,
            "mode": cfg.time_mode,
            "tolerance_h": cfg.time_tolerance_h,
            "tolerance_policy": "soft-warning",
            "chosen_time": sel_logic.chosen_time,
            "out_of_tolerance": tol_note,
        },
        "reference": {
            "design_id": cfg.reference.design_id,
            "design_id_resolved": ref_raw,
            "stat": cfg.reference.stat,
            "anchors": _anchors_payload(
                compute_reference_table(
                    sel_int.per_corner,
                    design_by=cfg.design_by,
                    ref_design_id=ref_raw,
                    stat=cfg.reference.stat,
                )
                if ref_raw is not None
                else None,
            ),
        },
        "eps": {
            "ratio": cfg.eps_ratio,
            "range": cfg.eps_range,
            "eta": cfg.eps_range,
            "ref": cfg.eps_ref,
            "abs": cfg.eps_abs,
        },
        "semantics": {
            "v": {
                "source_channel": cfg.response.logic_channel,
                "transform": "u = log2(L); v = (u - min(u))/((max(u)-min(u)) + eta)",
                "persisted_units": "unit-interval [0,1] (not log)",
                "eta": cfg.eps_range,
            },
            "y_star": {
                "source_channel": cfg.response.intensity_channel,
                "transform": "y* = log2( ((I + eps_abs)/max(A + alpha, eps_ref)) + delta )",
                "persisted_units": "log2",
                "alpha": cfg.ref_add_alpha,
                "delta": cfg.log2_offset_delta,
            },
            "r_logic": {
                "definition": "per design dynamic range on LOGIC (linear, ε-guarded): max(L_i)/min(L_i)",
                "epsilon_guard_ratio": cfg.eps_ratio,
                "stats_over_vec8": r_desc,
            },
        },
        "rows": {
            "per_corner_logic": int(len(sel_logic.per_corner)),
            "per_corner_intensity": int(len(sel_int.per_corner)),
            "points_logic": int(len(sel_logic.points)),
            "points_intensity": int(len(sel_int.points)),
            "vec8": int(len(vec8_out)),
        },
        "flat_logic_count": int(flat_count),
        "flat_logic_fraction": float(flat_fraction),
        "flat_logic_sample_design_ids": flat_samples,
    }

    return SFXIBuildResult(
        vec8=vec8_out,
        log=log_payload,
        cfg=cfg,
        sel_logic=sel_logic,
        sel_int=sel_int,
        ref_design_id=ref_raw,
    )


def run_sfxi(tidy_df: pd.DataFrame, xform_cfg: Any, out_dir: Path | str) -> None:
    """
    Called by reader.main. Consumes the in-memory tidy table and the YAML XForm
    block (or XForm object), writes vec8 + log to disk inside out_dir/sfxi/.
    """
    result = build_vec8_from_tidy(tidy_df, xform_cfg)

    # Emit soft warnings (once, from logic selection)
    if result.sel_logic.time_warning:
        warnings.warn(f"SFXI: {result.sel_logic.time_warning}", stacklevel=2)

    # Emit flat-logic warning (aggregate, per run)
    if "flat_logic" in result.vec8.columns and len(result.vec8) > 0:
        flat_mask = result.vec8["flat_logic"].astype(bool)
        flat_count = int(flat_mask.sum())
        if flat_count > 0:
            label_col = result.cfg.design_by[0] if result.cfg.design_by else None
            sample = []
            if label_col and label_col in result.vec8.columns:
                sample = result.vec8.loc[flat_mask, label_col].astype(str).dropna().drop_duplicates().head(5).tolist()
            frac = float(flat_count) / float(len(result.vec8)) if len(result.vec8) else 0.0
            sample_note = f" Sample design_ids: {', '.join(sample)}." if sample else ""
            warnings.warn(
                f"SFXI: flat logic detected for {flat_count}/{len(result.vec8)} designs ({frac:.1%}).{sample_note}",
                stacklevel=2,
            )

    # Write
    cfg = result.cfg
    write_outputs(
        vec8=result.vec8,
        log=result.log,
        out_dir=out_dir,
        subdir=cfg.output_subdir,
        vec8_filename=(f"{cfg.filename_prefix}_{cfg.vec8_filename}" if cfg.filename_prefix else cfg.vec8_filename),
        log_filename=(f"{cfg.filename_prefix}_{cfg.log_filename}" if cfg.filename_prefix else cfg.log_filename),
    )
