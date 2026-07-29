from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, cols: Iterable[str], *, where: str) -> None:
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns: {missing}")


def warn_if_empty(df: pd.DataFrame, *, where: str, detail: str | None = None) -> bool:
    if not df.empty:
        return False
    message = f"[warn]{where}[/warn] • no rows to plot"
    if detail:
        message += f" ({detail})"
    logging.getLogger("reader").info(message)
    return True


def alias_column(df: pd.DataFrame, name: str | None, suffix: str = "_alias") -> str | None:
    if name is None:
        return None
    candidate = f"{str(name)}{suffix}"
    return candidate if candidate in df.columns else name
