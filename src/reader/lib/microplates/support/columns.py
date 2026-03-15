"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/microplates/support/columns.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd


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


def alias_column(df: pd.DataFrame, name: str | None, suffix: str = "_alias") -> str | None:
    if name is None:
        return None
    candidate = f"{str(name)}{suffix}"
    return candidate if candidate in df.columns else name


def pretty_name(name: str, suffix: str = "_alias") -> str:
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name
