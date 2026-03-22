from __future__ import annotations

from collections.abc import Callable

from reader.plotting.sinks import PlotFigure

from .decomposition import render_decomposition_view
from .heatmap import render_library_heatmap_view
from .interaction import render_interaction_summary_view
from .pareto import render_pareto_view
from .shared import _RetronSummaryPlotRequest
from .stress_modulation import render_stress_modulation_view

SummaryViewRenderer = Callable[[_RetronSummaryPlotRequest], list[PlotFigure]]

SUMMARY_VIEW_RENDERERS: dict[str, SummaryViewRenderer] = {
    "interaction": render_interaction_summary_view,
    "heatmap": render_library_heatmap_view,
    "stress_modulation": render_stress_modulation_view,
    "decomposition": render_decomposition_view,
    "pareto": render_pareto_view,
}

__all__ = ["SUMMARY_VIEW_RENDERERS", "SummaryViewRenderer", "_RetronSummaryPlotRequest"]
