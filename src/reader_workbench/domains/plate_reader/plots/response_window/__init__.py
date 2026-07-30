"""Canonical response-window plot selection and rendering."""

from .diagnostic import (
    ResponseWindowDiagnostic,
    prepare_response_window_diagnostic,
)
from .diagnostic_render import render_response_window_diagnostic
from .schema import COMPONENT_COLUMNS, MAGNITUDE_COLUMNS, RESPONSE_COLUMNS, STATE_ORDER
from .summary import (
    ResponseWindowSummaryMatrix,
    render_response_window_summary,
    response_window_summary_matrix,
)

__all__ = [
    "COMPONENT_COLUMNS",
    "MAGNITUDE_COLUMNS",
    "RESPONSE_COLUMNS",
    "ResponseWindowDiagnostic",
    "ResponseWindowSummaryMatrix",
    "STATE_ORDER",
    "prepare_response_window_diagnostic",
    "render_response_window_diagnostic",
    "render_response_window_summary",
    "response_window_summary_matrix",
]
