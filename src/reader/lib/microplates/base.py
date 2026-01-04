"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/microplates/base.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

GroupMatch = Literal["exact", "contains", "startswith", "endswith", "regex"]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slugify(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return re.sub(r"_{2,}", "_", s).strip("_")


def save_figure(fig, output_dir: Path, filename_stub: str, ext: str = "pdf", dpi: int | None = None) -> Path:
    """
    Save figures as **PDF by default** (print-friendly, vector). Use ext="png" if you
    explicitly need rasters.
    """
    ensure_dir(output_dir)
    out = output_dir / f"{slugify(filename_stub)}.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    return out


def require_columns(df: pd.DataFrame, cols: Iterable[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns: {missing}")


def warn_if_empty(df: pd.DataFrame, *, where: str, detail: str | None = None) -> bool:
    if df.empty:
        msg = f"[warn]{where}[/warn] • no rows to plot"
        if detail:
            msg += f" ({detail})"
        logging.getLogger("reader").info(msg)
        return True
    return False


# -------------------- alias helpers --------------------


def alias_column(df: pd.DataFrame, name: str | None, suffix: str = "_alias") -> str | None:
    """
    Prefer '<name>_alias' when present; otherwise return 'name' unchanged.
    If name is None, return None. This is a deterministic preference, not a fallback.
    """
    if name is None:
        return None
    cand = f"{str(name)}{suffix}"
    return cand if cand in df.columns else name


def pretty_name(name: str, suffix: str = "_alias") -> str:
    """Strip a trailing alias suffix from a label for display purposes."""
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _match_value(val: str, needle: str, mode: GroupMatch) -> bool:
    v = str(val)
    n = str(needle)
    if mode == "exact":
        return v == n
    if mode == "contains":
        return n in v
    if mode == "startswith":
        return v.startswith(n)
    if mode == "endswith":
        return v.endswith(n)
    if mode == "regex":
        return re.search(n, v) is not None
    raise ValueError(f"Unknown group_match: {mode}")


def resolve_groups(
    universe: Iterable[str], groups: list[dict[str, list[str]]] | None, match: GroupMatch
) -> list[tuple[str, list[str]]]:
    """
    Expand a list like:
      - { "Group A": ["foo", "bar"] }
      - { "Group B": ["baz"] }
    into [("Group A", [...actual matches...]), ...]

    group_match semantics:
      - exact      : v == needle
      - contains   : needle in v
      - startswith : v starts with needle
      - endswith   : v ends with needle
      - regex      : re.search(needle, v) is not None
    """
    values = list(map(str, universe))
    if not groups:
        # single pass-through group
        return [("all", values)]
    resolved: list[tuple[str, list[str]]] = []
    for item in groups:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("groups must be list of single-key dicts")
        label, needles = next(iter(item.items()))
        picked: list[str] = []
        for n in needles:
            for v in values:
                if _match_value(v, n, match) and v not in picked:
                    picked.append(v)
        if picked:
            resolved.append((label, picked))
    return resolved


def nearest_time_per_key(
    df: pd.DataFrame,
    *,
    target_time: float,
    keys: Sequence[str],
    tol: float = 0.51,
    time_col: str = "time",
) -> pd.DataFrame:
    """
    For each key combination, choose the row at time nearest to target_time.
    Rows with |dt|>tol are dropped.
    """
    w = df.copy()
    w[time_col] = pd.to_numeric(w[time_col], errors="coerce")
    w = w.dropna(subset=[time_col])
    w["__dt__"] = (w[time_col] - float(target_time)).abs()
    idx = w.groupby(list(keys))["__dt__"].idxmin()
    picked = w.loc[idx].copy()
    return picked.loc[picked["__dt__"] <= float(tol)].drop(columns="__dt__")


# -------------------- subplot layout + smart string ordering --------------------


def best_subplot_grid(n: int) -> tuple[int, int]:
    """
    Choose near-square grid for n panels.
    """
    n = max(1, int(n))
    rows = int(math.floor(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    return rows, cols


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_first_number(s: str) -> float | None:
    m = _NUM_RE.search(str(s))
    return float(m.group(0)) if m else None


def _unit_scale_to_uM(s: str) -> float:
    """
    Rough unit detection → scale to µM for ordering.
    Recognizes: nM, uM (or µM), mM, M. If multiple present, first match wins.
    """
    t = str(s).lower()
    if "nm" in t:
        return 1e-3
    if "µm" in t or "um" in t or "μm" in t:
        return 1.0
    if "mm" in t:
        return 1e3
    # bare 'm' (molar) last to avoid catching 'mm'
    if re.search(r"\b(?<![a-z])m\b", t):
        return 1e6
    return 1.0


def smart_string_numeric_key(s: str) -> tuple[int, float, str]:
    """
    Ordering key that prefers numeric content (scaled by detected unit).
    - Returns (0, value_in_uM, lowercased_str) when a number is found,
      else (1, +inf, lowercased_str) so text-only items sort after numerics.
    """
    s_str = str(s)
    num = _extract_first_number(s_str)
    if num is None:
        return (1, float("inf"), s_str.lower())
    scale = _unit_scale_to_uM(s_str)
    return (0, float(num) * scale, s_str.lower())


def _prefix_before_number(s: str) -> str:
    """
    Return the normalized text before the first numeric token.
    Used to group conditions like 'IPTG 0 uM', 'IPTG 5 uM', 'IPTG 60 uM' together.
    """
    s_str = str(s)
    m = _NUM_RE.search(s_str)
    prefix = s_str[: m.start()].strip() if m else s_str.strip()
    # normalize whitespace only; keep punctuation (e.g., '+' in 'x + y')
    prefix = re.sub(r"\s+", " ", prefix).strip().lower()
    return prefix


def smart_grouped_dose_key(s: str) -> tuple[str, int, float, str]:
    """
    Ordering key that *preserves the non-numeric prefix* and sorts by dose within it.
    Examples:
      'IPTG 0 uM' < 'IPTG 5 uM' < 'IPTG 60 uM' < 'Arabinose 0 uM' < ...
    Units are normalized to µM for comparability (nM<uM<mM<M).
    Non-numeric entries within the same prefix sort after numeric ones.
    """
    s_str = str(s)
    prefix = _prefix_before_number(s_str)
    num = _extract_first_number(s_str)
    has_num = 0 if num is not None else 1
    scale = _unit_scale_to_uM(s_str)
    val = float(num) * scale if num is not None else float("inf")
    return (prefix, has_num, val, s_str.lower())
