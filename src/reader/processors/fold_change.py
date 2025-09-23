"""
--------------------------------------------------------------------------------
<reader project>
reader/processors/fold_change.py

Fold-change transform (registered).

Computes value := value / baseline for rows matching `on_channels`, grouped by
`by` keys, with a per-group baseline selected via either:

    • control_query: pandas query string evaluated on the group
      (e.g., "treatment == 'EtOH 0%, 0 nM'")

    • control_label: {column: value} exact match inside the group
      (e.g., {"treatment": "EtOH 0%, 0 nM"})

Writes new rows with channel renamed to "<channel>/FC" unless `rename_to` is set.

YAML examples:
    - type: fold_change
      by: ["genotype", "batch", "time"]          # group scope for baseline
      on_channels: ["YFP", "CFP", "YFP/CFP"]     # which channels to convert
      control_label: {treatment: "EtOH 0%, 0 nM"}
      rename_suffix: "/FC"

    - type: fold_change
      by: ["genotype", "batch"]
      on_channels: ["CFP/OD600"]
      control_query: "treatment.str.contains('control', case=False)"
      rename_to: "CFP/OD600_fc"

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from reader.config import ReaderCfg, XForm

from . import TransformContext, register_transform

LOG = logging.getLogger(__name__)


def _pick_baseline(g: pd.DataFrame, query: Optional[str], label: Optional[Dict[str, object]]) -> Optional[float]:
    """Return baseline scalar for group `g` (median over matched rows)."""
    sub = g
    if query:
        try:
            sub = g.query(query, engine="python")
        except Exception as e:
            raise ValueError(f"fold_change control_query error: {e}") from e
    if label:
        for k, v in label.items():
            sub = sub[sub[k].astype(str) == str(v)]
    if sub.empty:
        return None
    val = pd.to_numeric(sub["value"], errors="coerce").dropna()
    if val.empty:
        return None
    return float(np.median(val))


@register_transform("fold_change")
def transform_fold_change(df: pd.DataFrame, cfg: XForm, ctx: TransformContext, reader_cfg: ReaderCfg) -> pd.DataFrame:
    """
    Apply fold-change per configured scope. Produces NEW rows (non-destructive).
    """
    by = list(getattr(cfg, "by", [])) or ["genotype", "batch"]  # sensible default
    on_channels = list(getattr(cfg, "on_channels", []))
    if not on_channels:
        # If omitted, apply to all present channels
        on_channels = sorted(df["channel"].astype(str).unique().tolist())

    control_query = getattr(cfg, "control_query", None)
    control_label = getattr(cfg, "control_label", None)
    rename_suffix = getattr(cfg, "rename_suffix", "/FC")
    rename_to     = getattr(cfg, "rename_to", None)

    if control_query is None and control_label is None:
        LOG.warning("fold_change: no control specified; pass-through (no new rows)")
        return df

    working = df.copy()
    working["value"] = pd.to_numeric(working["value"], errors="coerce")

    parts: List[pd.DataFrame] = []
    kept = working[working["channel"].isin(on_channels)].copy()

    group_cols = [c for c in by if c in kept.columns]
    if not group_cols:
        group_cols = []

    for keys, g in kept.groupby(group_cols, dropna=False):
        baseline = _pick_baseline(g, control_query, control_label)
        if baseline is None or baseline == 0 or not np.isfinite(baseline):
            # Skip group if no valid baseline
            continue
        new = g.copy()
        new["value"] = new["value"] / float(baseline)
        if rename_to:
            # every channel → rename_to (single)
            new["channel"] = str(rename_to)
        else:
            new["channel"] = new["channel"].astype(str) + str(rename_suffix)
        parts.append(new)

    if not parts:
        LOG.warning("fold_change: produced no rows (no valid baselines found)")
        return df

    out = pd.concat([df] + parts, ignore_index=True)
    LOG.info("✓ fold_change: groups=%d, new_rows=%d", len(parts), sum(len(p) for p in parts))
    return out
