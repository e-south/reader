"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/logic_symmetry/encodings.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    # Preferred palette provider if available in reader
    from reader.utils.plot_style import PaletteBook  # type: ignore
except Exception:  # pragma: no cover
    PaletteBook = None  # fallback handled below


@dataclass(frozen=True)
class EncodingConfig:
    size_by: str           # "log_r" | "cv" | "fixed"
    size_fixed: float
    hue: Optional[str]     # None | column name
    alpha_by: Optional[str]
    alpha_min: float
    alpha_max: float
    shape_by: Optional[str]
    shape_cycle: List[str]
    shape_max_categories: Optional[int]


def _scale_to_range(values: pd.Series, vmin: float, vmax: float, lo: float, hi: float) -> pd.Series:
    if values.empty or np.all(values == values.iloc[0]):
        return pd.Series([lo] * len(values), index=values.index, dtype=float)
    # guard inf/nan
    s = values.replace([np.inf, -np.inf], np.nan).fillna(vmin)
    s = s.clip(lower=vmin, upper=vmax)
    norm = (s - vmin) / max(vmax - vmin, 1e-12)
    return lo + norm * (hi - lo)


def _compute_size(df: pd.DataFrame, cfg: EncodingConfig) -> pd.Series:
    if cfg.size_by == "fixed":
        return pd.Series([float(cfg.size_fixed)] * len(df), index=df.index, dtype=float)
    elif cfg.size_by == "log_r":
        # Robustly scale log_r into visually distinct point areas
        v = df["log_r"].clip(lower=0.0)
        vmax = float(np.nanpercentile(v, 95)) if len(v) else 1.0
        return _scale_to_range(v, 0.0, max(vmax, 1e-6), 40.0, 300.0)
    elif cfg.size_by == "cv":
        v = df["cv"].clip(lower=0.0)
        vmax = float(np.nanpercentile(v, 95)) if len(v) else 1.0
        return _scale_to_range(v, 0.0, max(vmax, 1e-6), 40.0, 300.0)
    raise ValueError(f"Unknown size_by '{cfg.size_by}'")


def _compute_alpha(df: pd.DataFrame, cfg: EncodingConfig) -> pd.Series:
    if not cfg.alpha_by:
        return pd.Series([cfg.alpha_max] * len(df), index=df.index, dtype=float)
    col = cfg.alpha_by
    if col not in df.columns:
        raise ValueError(f"alpha_by refers to missing column '{col}'")
    # Order categories numerically if possible (batch typically numeric)
    series = df[col]
    try:
        order = np.sort(series.astype(float).unique())
        order = [float(x) for x in order]
        order_map = {v: i for i, v in enumerate(order)}
        keys = series.astype(float).map(order_map)
    except Exception:
        cats = pd.Categorical(series.astype(str))
        keys = pd.Series(cats.codes, index=series.index)  # -1 for NaN
    kmin, kmax = float(keys.min()), float(keys.max())
    return _scale_to_range(keys.astype(float), kmin, kmax, cfg.alpha_min, cfg.alpha_max)


def _compute_shape(df: pd.DataFrame, cfg: EncodingConfig) -> pd.Series:
    if not cfg.shape_by:
        return pd.Series(["o"] * len(df), index=df.index, dtype=object)
    col = cfg.shape_by
    if col not in df.columns:
        raise ValueError(f"shape_by refers to missing column '{col}'")
    cats = pd.Categorical(df[col].astype(str))
    ncat = int(len(cats.categories))
    if cfg.shape_max_categories is not None and ncat > int(cfg.shape_max_categories):
        raise ValueError(f"shape_by={col!r} has {ncat} categories, exceeding shape_max_categories={cfg.shape_max_categories}")
    if ncat > len(cfg.shape_cycle):
        raise ValueError(f"Not enough markers in shape_cycle ({len(cfg.shape_cycle)}) for {ncat} categories; extend the cycle or lower categories.")
    mapping = {cat: cfg.shape_cycle[i] for i, cat in enumerate(cats.categories)}
    return pd.Series([mapping[str(v)] for v in cats.astype(str)], index=df.index, dtype=object)


def apply_encodings(
    df_points: pd.DataFrame,
    cfg: EncodingConfig
) -> pd.DataFrame:
    out = df_points.copy()
    out["size_value"] = _compute_size(out, cfg)
    out["alpha_value"] = _compute_alpha(out, cfg)
    out["hue_value"] = (out[cfg.hue] if cfg.hue else pd.Series([None]*len(out), index=out.index)).astype(object)
    out["shape_value"] = _compute_shape(out, cfg)
    return out
