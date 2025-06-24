"""
--------------------------------------------------------------------------------
<reader project>
src/reader/processors/fold_change.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

# ───────── helpers ────────────────────────────────────────────────────────
def _norm(s: str | None) -> str:
    s = (s or "").strip().replace("µ", "u")
    return re.sub(r"\s+", " ", s).casefold()

def _cfg_get(cfg: Any, key: str, default=None):
    return getattr(cfg, key, cfg.get(key, default) if isinstance(cfg, dict) else default)

def _select_snapshot(df: pd.DataFrame, times: Optional[List[float]], tol=0.51):
    if not times:
        return df
    avail = np.array(sorted(df["time"].unique()))
    keep  = []
    for rt in times:
        nearest = float(avail[np.abs(avail-rt).argmin()])
        if abs(nearest-rt) > tol:
            raise ValueError(f"Requested {rt} h but nothing within ±{tol} h")
        if nearest != rt:
            logger.info("Using %.2f h instead of %.2f h", nearest, rt)
        keep.append(nearest)
    return df[df["time"].isin(keep)].copy()

# ───────── FC core ────────────────────────────────────────────────────────
def _compute_fc_block(d: pd.DataFrame, *, baseline_label: str,
                      key_cols: list[str], fc_name: str, log2_name: str):
    d = d.copy()
    d["reference"]   = baseline_label
    d["_norm_treat"] = d["treatment"].map(_norm)

    base_rows = d[d["_norm_treat"] == _norm(baseline_label)]
    if base_rows.empty:
        raise ValueError(f"Baseline {baseline_label!r} missing in subset")

    baseline = (base_rows
                .groupby(key_cols+["time"])["value"]
                .mean()
                .rename("baseline")
                .reset_index())

    merged = d.merge(baseline, on=key_cols+["time"], how="left",
                     validate="many_to_one")
    if merged["baseline"].isna().any():
        raise RuntimeError("Baseline assignment failed")

    merged[fc_name]   = merged["value"] / merged["baseline"]
    merged[log2_name] = np.log2(merged[fc_name])
    return merged.drop(columns="_norm_treat")

# ───────── public API ─────────────────────────────────────────────────────
def apply_fold_change(df: pd.DataFrame, fc_cfg, *, out_dir: Path) -> pd.DataFrame:
    target       = _cfg_get(fc_cfg, "target")
    treat_col    = _cfg_get(fc_cfg, "treatment_column")
    report_times = _cfg_get(fc_cfg, "report_times")
    if isinstance(report_times, (int, float)):
        report_times = [float(report_times)]
    if treat_col != "treatment":
        raise NotImplementedError("Expecting a column named 'treatment'")

    # ── resolve column-name templates ------------------------------------
    out_cfg         = _cfg_get(fc_cfg, "output", {}) or {}
    fc_name_tpl     = out_cfg.get("fc_column",     "FC_{target}")
    log2_name_tpl   = out_cfg.get("log2fc_column", "log2FC_{target}")
    fc_name         = fc_name_tpl.format(target=target)
    log2_name       = log2_name_tpl.format(target=target)

    # ── source rows -------------------------------------------------------
    df_target = df[df["channel"] == target]
    if df_target.empty:
        raise ValueError(f"Channel '{target}' absent")
    snap      = _select_snapshot(df_target, report_times)

    key_cols  = _cfg_get(fc_cfg, "group_by") or []
    fc_blocks = []

    if _cfg_get(fc_cfg, "use_global_baseline", True):
        logger.info("Fold-change: GLOBAL baseline %r",
                    _cfg_get(fc_cfg, "global_baseline_value"))
        fc_blocks.append(_compute_fc_block(
            snap,
            baseline_label=_cfg_get(fc_cfg, "global_baseline_value"),
            key_cols=key_cols, fc_name=fc_name, log2_name=log2_name))
    else:
        grp_key = key_cols[0] if key_cols else "genotype"
        for ov in _cfg_get(fc_cfg, "overrides", []):
            sub = snap[snap[grp_key] == ov.get(grp_key)]
            if sub.empty:
                logger.warning("No rows for %s=%s – skipping", grp_key, ov.get(grp_key))
                continue
            logger.info("Fold-change: %s  (reference=%r)", ov.get(grp_key),
                        ov.get("baseline_value"))
            fc_blocks.append(_compute_fc_block(
                sub,
                baseline_label=ov.get("baseline_value"),
                key_cols=key_cols, fc_name=fc_name, log2_name=log2_name))

    if not fc_blocks:
        raise RuntimeError("No fold-change data generated")

    fc_df = pd.concat(fc_blocks, ignore_index=True)

    # ── summary CSV + verbose console lines ------------------------------
    summary_cols = (key_cols or []) + ["time", "reference", "treatment"]
    summary = (fc_df[summary_cols + [fc_name, log2_name]]
               .groupby(summary_cols, as_index=False)
               .mean())
    for _, r in summary.iterrows():
        info = ", ".join(f"{c}={r[c]!r}" for c in summary_cols)
        logger.info("   ↳ %s → %s = %.3fx  (log2=%.2f)",
                    info, fc_name, r[fc_name], r[log2_name])

    (out_dir / "fold_change_summary.csv").write_text(summary.to_csv(index=False))
    logger.info("✓ wrote fold-change summary → fold_change_summary.csv")

    # ── LONG-FORM melt so FC & log2FC become channels --------------------
    meta_cols = [c for c in df.columns if c not in ("channel", "value")]
    long_fc = (fc_df
               .drop(columns=["value", "baseline"], errors="ignore")
               .melt(id_vars=meta_cols,
                     value_vars=[fc_name, log2_name],
                     var_name="channel",
                     value_name="value"))

    return pd.concat([df, long_fc[meta_cols + ["channel", "value"]]],
                     ignore_index=True)