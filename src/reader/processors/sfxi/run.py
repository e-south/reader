"""
--------------------------------------------------------------------------------
<reader project>
reader/processors/sfxi/run.py

Program entry for SFXI: tidy_data → vec8

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .api import load_sfxi_config
from .math import compute_vec8
from .selection import cornerize_and_aggregate
from .writer import write_outputs


def _assert_same_times(chosen_a: Dict[object, float], chosen_b: Dict[object, float]) -> None:
    if set(chosen_a.keys()) != set(chosen_b.keys()):
        raise ValueError("SFXI: logic and intensity selections produced different batch sets.")
    for k in chosen_a:
        ta, tb = float(chosen_a[k]), float(chosen_b[k])
        if not np.isclose(ta, tb, rtol=0, atol=1e-9):
            raise ValueError(f"SFXI: logic and intensity channels selected different times for batch={k!r}: {ta} vs {tb}")


def _attach_sequence(vec8: pd.DataFrame, tidy_df: pd.DataFrame, *, design_by: List[str], batch_col: str) -> pd.DataFrame:
    # If vec8 already has a usable 'sequence' column, keep it.
    if "sequence" in vec8.columns and vec8["sequence"].notna().any():
        return vec8

    if "sequence" not in tidy_df.columns:
        return vec8

    label_col = design_by[0]
    if not {label_col, batch_col, "sequence"}.issubset(tidy_df.columns):
        return vec8

    seq_map = (
        tidy_df[[label_col, batch_col, "sequence"]]
        .dropna(subset=["sequence"])
        .drop_duplicates(subset=[label_col, batch_col], keep="first")
    )

    merged = vec8.merge(seq_map, on=[label_col, batch_col], how="left", suffixes=("", "_ref"))
    # Prefer existing 'sequence' if present; otherwise use attached value
    if "sequence_ref" in merged.columns:
        if "sequence" not in merged.columns:
            merged["sequence"] = merged["sequence_ref"]
        else:
            merged["sequence"] = merged["sequence"].fillna(merged["sequence_ref"])
        merged = merged.drop(columns=["sequence_ref"])

    return merged


def _reorder_and_filter(vec8: pd.DataFrame, *, design_by: List[str], batch_col: str, ref_genotype: str | None) -> pd.DataFrame:
    """
    Apply user-requested:
      - drop REF genotype rows
      - reorder columns to:
        genotype, sequence, r_logic, v00, v10, v01, v11,
        y00_star, y10_star, y01_star, y11_star, flat_logic
    """
    label_col = design_by[0]

    df = vec8.copy()

    # Exclude REF (if specified)
    if ref_genotype is not None and label_col in df.columns:
        df = df[df[label_col].astype(str) != str(ref_genotype)].copy()

    # Ensure 'sequence' exists even if missing upstream
    if "sequence" not in df.columns:
        df["sequence"] = pd.NA

    # Build final column order; tolerate absence of label_col under alias 'genotype'
    # We will expose the label column as 'genotype' in output for clarity.
    # If label_col is already 'genotype', this is a no-op rename.
    if label_col != "genotype" and label_col in df.columns:
        df = df.rename(columns={label_col: "genotype"})
    elif "genotype" not in df.columns and label_col in df.columns:
        df["genotype"] = df[label_col]

    desired = [
        "genotype",
        "sequence",
        "r_logic",
        "v00", "v10", "v01", "v11",
        "y00_star", "y10_star", "y01_star", "y11_star",
        "flat_logic",
    ]

    # Keep only desired columns in that order (ignore extras like batch)
    existing = [c for c in desired if c in df.columns]
    out = df[existing].copy()

    # Final tidy: drop duplicates just in case and reset index
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def run_sfxi(tidy_df: pd.DataFrame, xform_cfg: Any, out_dir: Path | str) -> None:
    """
    Called by reader.main. Consumes the in-memory tidy table and the YAML XForm
    block (or XForm object), writes vec8 + log to disk inside out_dir/sfxi/.
    """
    cfg = load_sfxi_config(xform_cfg)

    # 1) Cornerize + aggregate (LOGIC channel)
    sel_logic = cornerize_and_aggregate(
        tidy_df,
        design_by=cfg.design_by,
        batch_col=cfg.batch_col,
        treatment_map=cfg.treatment_map,
        case_sensitive=cfg.treatment_case_sensitive,
        time_column=cfg.time_column,
        channel=cfg.response.logic_channel,
        target_time_h=cfg.target_time_h,
        time_mode=cfg.time_mode,
        time_tolerance_h=cfg.time_tolerance_h,
        time_per_batch=cfg.time_per_batch,
        on_missing_time=cfg.on_missing_time,
        require_all_corners_per_design=cfg.require_all_corners_per_design,
    )

    # 2) Cornerize + aggregate (INTENSITY channel)
    sel_int = cornerize_and_aggregate(
        tidy_df,
        design_by=cfg.design_by,
        batch_col=cfg.batch_col,
        treatment_map=cfg.treatment_map,
        case_sensitive=cfg.treatment_case_sensitive,
        time_column=cfg.time_column,
        channel=cfg.response.intensity_channel,
        target_time_h=cfg.target_time_h,
        time_mode=cfg.time_mode,
        time_tolerance_h=cfg.time_tolerance_h,
        time_per_batch=cfg.time_per_batch,
        on_missing_time=cfg.on_missing_time,
        require_all_corners_per_design=cfg.require_all_corners_per_design,
    )

    # 3) Safety: same chosen snapshot times
    _assert_same_times(sel_logic.chosen_times, sel_int.chosen_times)

    # 4) Compute vec8 (logic from logic channel, y* from intensity channel + anchors)
    vec8 = compute_vec8(
        points_logic=sel_logic.points,
        points_intensity=sel_int.points,
        per_corner_intensity=sel_int.per_corner,
        design_by=cfg.design_by, batch_col=cfg.batch_col,
        reference_genotype=cfg.reference.genotype,
        reference_scope=cfg.reference.scope,
        reference_stat=cfg.reference.stat,
        eps_ratio=cfg.eps_ratio, eps_range=cfg.eps_range,
        eps_ref=cfg.eps_ref, eps_abs=cfg.eps_abs,
    )

    # 4b) Attach 'sequence' if available in the tidy inputs
    vec8 = _attach_sequence(vec8, tidy_df, design_by=cfg.design_by, batch_col=cfg.batch_col)

    # 4c) Apply requested filtering and column order
    vec8_out = _reorder_and_filter(
        vec8,
        design_by=cfg.design_by,
        batch_col=cfg.batch_col,
        ref_genotype=cfg.reference.genotype,
    )

    # 5) Compose a log payload (unchanged)
    log_payload: Dict[str, Any] = {
        "name": cfg.name,
        "design_by": cfg.design_by,
        "batch_col": cfg.batch_col,
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
            "per_batch": cfg.time_per_batch,
            "on_missing_time": cfg.on_missing_time,
            "chosen_times": sel_logic.chosen_times,  # same as sel_int
            "dropped_batches": sorted(set(sel_logic.dropped_batches) | set(sel_int.dropped_batches)),
        },
        "reference": {
            "genotype": cfg.reference.genotype,
            "scope": cfg.reference.scope,
            "stat": cfg.reference.stat,
        },
        "eps": {
            "ratio": cfg.eps_ratio, "range": cfg.eps_range,
            "ref": cfg.eps_ref, "abs": cfg.eps_abs,
        },
        "rows": {
            "per_corner_logic": int(len(sel_logic.per_corner)),
            "per_corner_intensity": int(len(sel_int.per_corner)),
            "points_logic": int(len(sel_logic.points)),
            "points_intensity": int(len(sel_int.points)),
            "vec8": int(len(vec8_out)),
        },
    }

    # 6) Write
    write_outputs(
        vec8=vec8_out, log=log_payload,
        out_dir=out_dir,
        subdir=cfg.output_subdir,
        vec8_filename=(f"{cfg.filename_prefix}_{cfg.vec8_filename}" if cfg.filename_prefix else cfg.vec8_filename),
        log_filename=(f"{cfg.filename_prefix}_{cfg.log_filename}" if cfg.filename_prefix else cfg.log_filename),
    )
