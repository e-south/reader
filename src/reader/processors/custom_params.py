"""
--------------------------------------------------------------------------------
<reader project>
reader/processors/custom_params.py

Common transforms:
    - ratio
    - blank_correction (also captures blanks for plotting)
    - overflow_handling
    - outlier_filter (no-op placeholder unless enabled)

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from textwrap import indent
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from reader.config import CustomParameter, ReaderCfg, XForm

from . import TransformContext, register_transform

LOG = logging.getLogger(__name__)


# ───────────────────────── helpers (shared) ─────────────────────────

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty DataFrame with the same columns as *df*."""
    return df.iloc[0:0].copy()


def _detect_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic extraction of blanks for plotting:
        • a boolean-ish 'is_blank' / 'blank' / 'isBlank'
        • OR 'treatment' / 'genotype' / 'sample_type' contains 'blank'
    Returns empty DataFrame (with same columns) if none found.
    """
    if df.empty:
        return _empty_like(df)

    cands = df.copy()
    mask = pd.Series(False, index=cands.index)

    for col in ["is_blank", "blank", "isBlank"]:
        if col in cands.columns:
            try:
                mask = mask | cands[col].astype(bool)
            except Exception:
                # tolerate weird types (e.g., strings "TRUE"/"FALSE")
                mask = mask | cands[col].astype(str).str.lower().isin({"true", "t", "1", "yes"})

    for col in ["treatment", "genotype", "sample_type"]:
        if col in cands.columns:
            mask = mask | cands[col].astype(str).str.contains("blank", case=False, na=False)

    return cands[mask].copy() if mask.any() else _empty_like(df)


# ─────────────────────────── ratio ─────────────────────────────────────────

def _build_ratio(df: pd.DataFrame, *, name: str, numerator: str, denominator: str) -> Tuple[pd.DataFrame, int]:
    """Return DataFrame with a new ratio channel appended and count of new rows."""
    if df.empty:
        return df, 0

    base_cols = [c for c in df.columns if c not in {"value"}]
    key_cols  = [c for c in base_cols if c in {"position", "time"}]

    lhs = df[df["channel"] == str(numerator)].copy()
    rhs = df[df["channel"] == str(denominator)].copy()
    if lhs.empty or rhs.empty:
        LOG.warning("ratio %s skipped: missing numerator (%s) or denominator (%s) channel",
                    name, numerator, denominator)
        return df, 0

    lhs = lhs.rename(columns={"value": "__num__"})
    rhs = rhs.rename(columns={"value": "__den__"})
    merged = pd.merge(lhs, rhs[key_cols + ["__den__"]], on=key_cols, how="inner")
    merged["value"]   = _safe_numeric(merged["__num__"]) / _safe_numeric(merged["__den__"])
    merged["channel"] = str(name)

    out = pd.concat([df, merged[base_cols + ["value"]]], ignore_index=True)
    return out, len(merged)


@register_transform("ratio")
def transform_ratio(df: pd.DataFrame, cfg: XForm, ctx: TransformContext, reader_cfg: ReaderCfg) -> pd.DataFrame:
    """
    Build a ratio channel: value(num) / value(den) at matching (position,time).
    YAML:
        - type: ratio
          name: "YFP/CFP"
          numerator: "YFP"
          denominator: "CFP"
    """
    name = getattr(cfg, "name", None)
    num  = getattr(cfg, "numerator", None)
    den  = getattr(cfg, "denominator", None)
    if not name or not num or not den:
        raise ValueError("ratio requires fields: name, numerator, denominator")

    out, n_new = _build_ratio(df, name=str(name), numerator=str(num), denominator=str(den))
    LOG.info("✓ built ratio %s = %s/%s → %d rows", name, num, den, n_new)
    return out


# ──────────────────────── blank correction ─────────────────────────────────

@register_transform("blank_correction")
def transform_blank_correction(df: pd.DataFrame, cfg: XForm, ctx: TransformContext, reader_cfg: ReaderCfg) -> pd.DataFrame:
    """
    Supported:
        method: "disregard" (default) – leave data unchanged
                "subtract"           – subtract per-channel median of blanks
        capture_blanks: true (default) – stash blanks in context for plotting
    """
    method  = str(getattr(cfg, "method", "disregard")).lower()
    capture = bool(getattr(cfg, "capture_blanks", True))

    # Capture blanks (for downstream plotting)
    if capture:
        blanks = _detect_blanks(df)
        try:
            ctx.blanks = blanks
        except Exception:
            pass
        LOG.debug("blank_correction: captured blanks → %d rows", len(blanks))

    if method == "disregard":
        LOG.info("✓ blank_correction (method=disregard)")
        return df

    if method == "subtract":
        blanks = getattr(ctx, "blanks", None)
        if blanks is None or blanks.empty:
            blanks = _detect_blanks(df)

        if blanks is None or blanks.empty:
            LOG.warning("blank_correction subtract requested but no blanks detected; no-op")
            return df

        corrections = (
            blanks.assign(value=_safe_numeric(blanks["value"]))
                  .groupby(["channel"])["value"].median()
                  .rename("__blank_median__")
        )
        out = df.copy()
        out["value"] = _safe_numeric(out["value"])
        out = out.join(corrections, on="channel")
        out["value"] = out["value"] - out["__blank_median__"].fillna(0.0)
        out = out.drop(columns=["__blank_median__"])
        LOG.info("✓ blank_correction (method=subtract) applied per channel")
        return out

    raise ValueError(f"Unknown blank_correction method '{method}'")


# ─────────────────────── overflow handling ─────────────────────────────────

def _clip_overflows(df: pd.DataFrame, *, channel_col: str, value_col: str, q: float = 0.999) -> pd.DataFrame:
    """
    Clip extreme/saturated readings per channel at the q-quantile
    and print a compact, human-friendly summary of what changed.
    """
    if df.empty:
        LOG.info("✓ overflow_handling: dataset empty; nothing to clip")
        return df

    # thresholds per channel
    thr = (
        df.groupby(channel_col)[value_col]
          .quantile(q)
          .rename("__clip_thr")
    )
    out = df.merge(thr, left_on=channel_col, right_index=True, how="left")

    # compute mask before mutating
    v = _safe_numeric(out[value_col])
    t = _safe_numeric(out["__clip_thr"])
    mask = v > t

    # do the clip
    out[value_col] = np.minimum(v, t)

    # succinct report (INFO)
    if bool(mask.any()):
        LOG.info("✓ overflow_handling: max clip at q=%.3f per channel", q)
        changed = out.loc[mask, [channel_col, "position", "time", value_col]].copy()
        changed = changed.rename(columns={value_col: "value_clipped"})

        # counts + examples per channel (cap to keep output short)
        counts = changed[channel_col].value_counts()
        LOG.info("• overflow summary: %d channels clipped; top offenders: %s",
                 counts.size,
                 ", ".join(f"{ch}={int(counts[ch])}" for ch in counts.head(5).index))

        # show a few examples per top channel
        for ch in counts.head(3).index:
            try:
                cthr = float(thr.loc[ch])
            except Exception:
                cthr = float("nan")
            sub = changed[changed[channel_col] == ch].head(5)
            # original values from pre-clip df
            orig = (
                df[df[channel_col] == ch][["position", "time", value_col]]
                  .rename(columns={value_col: "value_orig"})
            )
            eg = (
                sub.merge(orig, on=["position", "time"], how="left")
                   .loc[:, ["position", "time", "value_orig", "value_clipped"]]
            )
            LOG.info("  • %s: thr=%.6g, n_clipped=%d\n%s",
                     ch, cthr, int(counts[ch]),
                     indent(eg.to_string(index=False), "    "))
    else:
        LOG.info("✓ overflow_handling: no clipping needed at q=%.3f", q)

    return out.drop(columns="__clip_thr")


@register_transform("overflow_handling")
def transform_overflow(df: pd.DataFrame, cfg: XForm, ctx: TransformContext, reader_cfg: ReaderCfg) -> pd.DataFrame:
    """
    Handle infinities and extreme values per channel.

    YAML:
        - type: overflow_handling
          action: "max" | "drop" | "none" | "nan"
          clip_quantile: 0.999   # used when action='max'
    """
    action = str(getattr(cfg, "action", "max")).lower()
    q = float(getattr(cfg, "clip_quantile", 0.999))

    out = df.copy()
    out["value"] = _safe_numeric(out["value"])

    if action == "none":
        LOG.info("✓ overflow_handling: none")
        return out

    if action == "drop":
        before = len(out)
        out = out.dropna(subset=["value"])
        LOG.info("✓ overflow_handling: drop → removed %d rows", before - len(out))
        return out

    if action == "nan":
        # leave NaNs (and infs already coerced to NaN) in place
        LOG.info("✓ overflow_handling: convert non-numeric/inf → NaN (kept)")
        return out

    if action == "max":
        return _clip_overflows(out, channel_col="channel", value_col="value", q=q)

    raise ValueError(f"Unknown overflow action '{action}'")


# ───────────────────────── outlier filter ──────────────────────────────────

@register_transform("outlier_filter")
def transform_outlier_filter(df: pd.DataFrame, cfg: XForm, ctx: TransformContext, reader_cfg: ReaderCfg) -> pd.DataFrame:
    """
    Optional simple outlier filter by z-score per (channel, time).

    YAML:
        - type: outlier_filter
          enable: true
          z_thresh: 4.0
    """
    enable = bool(getattr(cfg, "enable", False))
    if not enable:
        LOG.info("✓ outlier_filter: disabled")
        return df

    z_th = float(getattr(cfg, "z_thresh", 4.0))
    out = df.copy()
    out["value"] = _safe_numeric(out["value"])

    def _f(g: pd.DataFrame) -> pd.DataFrame:
        mu = g["value"].mean()
        sd = g["value"].std(ddof=1) if g["value"].size > 1 else 0.0
        if not np.isfinite(sd) or sd <= 0:
            return g
        z = (g["value"] - mu) / sd
        return g.loc[z.abs() <= z_th]

    before = len(out)
    out = out.groupby(["channel", "time"], group_keys=False).apply(_f)
    LOG.info("✓ outlier_filter: z<=%.2f → removed %d rows", z_th, before - len(out))
    return out


# ─────────────────────── pipeline-style convenience ────────────────────────
# This function is what the top-level pipeline calls directly (see main.py).
# It mirrors a few common transforms succinctly without requiring YAML-driven
# transform sequencing. It also returns the detected blanks for plotting.

def apply_custom_parameters(
    df: pd.DataFrame,
    *,
    blank_correction: str = "disregard",
    overflow_action: str = "max",
    custom_parameters: Optional[Iterable[CustomParameter]] = None,
    outlier_filter: bool | dict = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Minimal, opinionated transform bundle used by the CLI:
      1) capture blanks (+ optional subtract)
      2) overflow handling (max/drop/none/nan)
      3) ratios (from `custom_parameters`)
      4) optional outlier filter

    Returns
    -------
    tidy_df, blanks_df
    """
    if df is None or df.empty:
        return df, _empty_like(df)

    # 1) blanks
    blanks = _detect_blanks(df)
    if blank_correction and str(blank_correction).lower() == "subtract":
        corr = (
            blanks.assign(value=_safe_numeric(blanks["value"]))
                  .groupby("channel")["value"].median()
                  .rename("__blank_median__")
        )
        df2 = df.copy()
        df2["value"] = _safe_numeric(df2["value"])
        df2 = df2.join(corr, on="channel")
        df2["value"] = df2["value"] - df2["__blank_median__"].fillna(0.0)
        df2 = df2.drop(columns=["__blank_median__"])
        LOG.info("✓ blank_correction (method=subtract) applied per channel")
    else:
        LOG.info("✓ blank_correction (method=disregard)")
        df2 = df.copy()
        df2["value"] = _safe_numeric(df2["value"])

    # 2) overflow
    action = str(overflow_action or "max").lower()
    if action == "max":
        df2 = _clip_overflows(df2, channel_col="channel", value_col="value", q=0.999)
    elif action == "drop":
        before = len(df2)
        df2 = df2.dropna(subset=["value"])
        LOG.info("✓ overflow_handling: drop → removed %d rows", before - len(df2))
    elif action == "nan":
        LOG.info("✓ overflow_handling: convert non-numeric/inf → NaN (kept)")
    elif action == "none":
        LOG.info("✓ overflow_handling: none")
    else:
        raise ValueError(f"Unknown overflow action '{overflow_action}'")

    # 3) ratios (CustomParameter objects with type='ratio' and parameters=[num, den])
    if custom_parameters:
        for cp in custom_parameters:
            if getattr(cp, "type", None) != "ratio":
                continue
            name = getattr(cp, "name", None)
            params = list(getattr(cp, "parameters", []))
            if not name or len(params) != 2:
                LOG.warning("Skipping custom parameter %r: need name + 2 parameters (num, den)", cp)
                continue
            df2, n_add = _build_ratio(df2, name=str(name), numerator=str(params[0]), denominator=str(params[1]))
            LOG.info("✓ built ratio %s = %s/%s → %d rows", name, params[0], params[1], n_add)

    # 4) optional outlier filter (same simple rule as transform_outlier_filter)
    if outlier_filter:
        z_th = 4.0
        if isinstance(outlier_filter, dict):
            z_th = float(outlier_filter.get("z_thresh", z_th))

        def _f(g: pd.DataFrame) -> pd.DataFrame:
            mu = g["value"].mean()
            sd = g["value"].std(ddof=1) if g["value"].size > 1 else 0.0
            if not np.isfinite(sd) or sd <= 0:
                return g
            z = (g["value"] - mu) / sd
            return g.loc[z.abs() <= z_th]

        before = len(df2)
        df2 = df2.groupby(["channel", "time"], group_keys=False).apply(_f)
        LOG.info("✓ outlier_filter: z<=%.2f → removed %d rows", z_th, before - len(df2))
    else:
        LOG.debug("outlier_filter: disabled")

    return df2, blanks

