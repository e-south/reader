"""
--------------------------------------------------------------------------------
<reader project>
src/reader/processors/custom_params.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd

from reader.config import CustomParameter

logger = logging.getLogger(__name__)


def apply_custom_parameters(
    df: pd.DataFrame,
    *,
    blank_correction: str = "avg_blank",
    overflow_action: str = "max",
    outlier_filter: bool = False,
    custom_parameters: Optional[List[CustomParameter]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    df
        Tidy long-form table containing *position* / *time* / *channel* / *value*
        plus any metadata columns produced by the parser + plate-map merge.
    blank_correction
        Strategy to subtract blanks – ``disregard``\ |``avg_blank``\ |``median_blank``\ |
        ``small_pos`` (≥ 0.001).
    overflow_action
        How to handle plate-reader “OVRFLW” entries – ``max``\ |``min``\ |``zero``\ |``drop``.
    outlier_filter
        Remove values outside μ ± 3 σ **per channel** *after* blank subtraction.
    custom_parameters
        List of :pyclass:`reader.config.CustomParameter` objects
        (currently only ``type="ratio"`` is supported).

    Returns
    -------
    data_corrected, blanks
    """
    # ────────────────────────────────── 1) split blanks ─────────────────────────────────
    blanks = df[df["type"] == "blank"].copy()
    data   = df[df["type"] != "blank"].copy()
    logger.info("Split into %d blanks and %d samples", len(blanks), len(data))

    # ─────────────────────────────── 2) overflow handling ───────────────────────────────
    overflow_mask = data["value"] == "OVRFLW"
    if overflow_mask.any():
        events = data.loc[overflow_mask, ["position", "time", "channel"]]
        logger.warning("Detected %d overflow readings:\n%s",
                       overflow_mask.sum(), events.to_string(index=False))

    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    if overflow_action == "zero":
        data["value"] = data["value"].fillna(0)
    elif overflow_action in ("max", "min"):
        fill_vals = data.groupby("channel")["value"].transform(overflow_action)
        data["value"] = data["value"].fillna(fill_vals)
    elif overflow_action == "drop":
        before = len(data)
        data = data.loc[~overflow_mask].copy()
        logger.info("Dropped %d overflow rows", before - len(data))
    else:
        raise ValueError(f"Unknown overflow_action '{overflow_action}'")
    logger.info("Overflow handled using '%s' strategy", overflow_action)

    # ─────────────────────────────── 3) blank subtraction ───────────────────────────────
    if blank_correction != "disregard" and not blanks.empty:
        blanks["value"] = pd.to_numeric(blanks["value"], errors="coerce")
        metric = (
            blanks.groupby("channel")["value"].mean()
            if blank_correction == "avg_blank"
            else blanks.groupby("channel")["value"].median()
        )

        logger.info("Computed blank metric (%s):\n%s",
                    blank_correction, metric.to_string())

        def _subtract(row):
            corrected = row["value"] - metric.get(row["channel"], 0.0)
            if blank_correction == "small_pos":
                return max(corrected, 1e-3)
            return max(corrected, 0)

        data["value"] = data.apply(_subtract, axis=1)
        logger.info("Applied '%s' blank subtraction", blank_correction)
    else:
        logger.info("Skipped blank subtraction (blank_correction='%s')",
                    blank_correction)

    # ─────────────────────────────── 4) outlier removal ────────────────────────────────
    if outlier_filter:

        def _rm_outliers(g):
            mu, sd = g["value"].mean(), g["value"].std()
            mask = (g["value"] >= mu - 3 * sd) & (g["value"] <= mu + 3 * sd)
            if (~mask).any():
                logger.info("Removed %d outliers in %s", (~mask).sum(), g.name)
            return g[mask]

        data = data.groupby("channel", group_keys=False).apply(_rm_outliers)

    # ─────────────────────────────── 5) de-duplicate ───────────────────────────────────
    before = len(data)
    data = data.drop_duplicates(subset=["position", "time", "channel"])
    if len(data) < before:
        logger.info("Dropped %d duplicate readings", before - len(data))

    # ─────────────────────────────── 6) custom ratios ──────────────────────────────────
    if custom_parameters:
        logger.info("Computing %d custom parameters via rounded-time exact match",
                    len(custom_parameters))
        derived: List[pd.DataFrame] = []

        # -------- time rounding so channels recorded a few seconds apart still align
        data["time_rnd"] = data["time"].round(3)      # 0.001 h ≈ 3.6 s

        has_sheet = "sheet_index" in data.columns
        index_cols = (["sheet_index", "time_rnd", "position"]
                      if has_sheet else ["time_rnd", "position"])

        # meta columns must exclude any index members
        meta_cols = [
            c for c in data.columns
            if c not in ("position", "time", "time_rnd", "channel", "value")
               and c not in index_cols
        ]

        wide = (
            data.pivot_table(index=index_cols + meta_cols,
                             columns="channel",
                             values="value",
                             aggfunc="first")
            .reset_index()
        )

        for cp in custom_parameters:
            if cp.type.lower() != "ratio" or len(cp.parameters) != 2:
                logger.warning("Skipping non-ratio CP: %s", cp)
                continue

            num, den = cp.parameters
            if num not in wide.columns or den not in wide.columns:
                logger.debug("Missing channels for %s – skipped", cp.name)
                continue

            out = wide[index_cols + meta_cols].copy()
            out = out.rename(columns={"time_rnd": "time"})  # restore canonical name
            out["channel"] = cp.name
            out["value"] = wide[num] / (wide[den] + 1e-6)
            derived.append(out)

        if derived:
            data = pd.concat([data, *derived], ignore_index=True)
            logger.info("Appended %d ratio rows", sum(len(d) for d in derived))

        # clean up helper column
        data.drop(columns="time_rnd", inplace=True, errors="ignore")

    return data, blanks