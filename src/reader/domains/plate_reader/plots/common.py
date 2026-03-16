"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/plots/common.py

Shared plotting helpers for plate-reader renderers.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd

from reader.core.plot_sinks import PlotFigure
from reader.core.plot_utils import save_figure

if TYPE_CHECKING:
    from reader.core.plot_style import PaletteBook


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


def best_subplot_grid(n: int) -> tuple[int, int]:
    n = max(1, int(n))
    rows = int(math.floor(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    return rows, cols


def colors_for(n: int, palette_book: PaletteBook | None) -> list[str]:
    if palette_book:
        if n == 1:
            palette = palette_book.colors(2)
            return [palette[1]] if (palette and str(palette[0]).lower() in {"#000000", "black"}) else [palette[0]]
        return palette_book.colors(n)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        raise RuntimeError("No color cycle available; configure Matplotlib rcParams or provide a PaletteBook.")
    if n == 1 and str(cycle[0]).lower() in {"#000000", "black"} and len(cycle) > 1:
        return [cycle[1]]
    return cycle[:n]


def emit_plot_figure(
    *,
    fig: Any,
    filename: str,
    output_dir: Path | None,
    fig_kwargs: Mapping[str, Any] | None,
) -> list[PlotFigure]:
    ext = str((fig_kwargs or {}).get("ext", "pdf")).lower()
    dpi = (fig_kwargs or {}).get("dpi", None)
    if output_dir is None:
        return [PlotFigure(fig=fig, filename=filename, ext=ext, dpi=dpi)]
    save_figure(fig, Path(output_dir), filename, ext=ext, dpi=dpi)
    plt.close(fig)
    return []


__all__ = [
    "alias_column",
    "best_subplot_grid",
    "colors_for",
    "emit_plot_figure",
    "pretty_name",
    "require_columns",
    "warn_if_empty",
]
