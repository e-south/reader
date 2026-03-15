"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/_shared.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from reader.core.plot_sinks import PlotFigure, normalize_plot_figures, save_plot_figures


def save_rendered_figures(*, ctx, figures: list[PlotFigure], plot_key: str) -> dict[str, list[str] | None]:
    normalized = normalize_plot_figures(figures, where=f"plot/{plot_key}")
    saved = save_plot_figures(normalized, ctx.plots_dir)
    return {"files": [str(path) for path in saved] if saved else None}
