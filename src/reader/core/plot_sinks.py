"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/plot_sinks.py

Plot sinks for saving rendered figures.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reader.core.errors import ExecutionError
from reader.lib.microplates.base import save_figure


@dataclass(frozen=True)
class PlotFigure:
    fig: Any
    filename: str
    ext: str = "pdf"
    dpi: int | None = None


def normalize_plot_figures(rendered: Any, *, where: str) -> list[PlotFigure]:
    if rendered is None:
        return []
    if isinstance(rendered, PlotFigure):
        return [rendered]
    if isinstance(rendered, Iterable) and not isinstance(rendered, str | bytes):
        figures: list[PlotFigure] = []
        for item in rendered:
            if not isinstance(item, PlotFigure):
                raise ExecutionError(f"{where}: render must return PlotFigure objects, got {type(item).__name__}")
            figures.append(item)
        return figures
    raise ExecutionError(f"{where}: render must return PlotFigure or list[PlotFigure], got {type(rendered).__name__}")


def save_plot_figures(figures: list[PlotFigure], output_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if not figures:
        return saved
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ExecutionError("Plot saving requires matplotlib.") from exc
    seen_figs: list[Any] = []
    for item in figures:
        ext = str(item.ext or "pdf").lstrip(".").lower()
        if not item.filename or not str(item.filename).strip():
            raise ExecutionError("PlotFigure.filename must be a non-empty string")
        path = save_figure(item.fig, output_dir, str(item.filename), ext=ext, dpi=item.dpi)
        saved.append(path)
        if item.fig not in seen_figs:
            seen_figs.append(item.fig)
    for fig in seen_figs:
        with suppress(Exception):
            plt.close(fig)
    return saved
