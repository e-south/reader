"""Canonical four-state event-window plot selection and rendering."""

from .diagnostic import (
    FourStateEventWindowDiagnostic,
    prepare_four_state_event_window_diagnostic,
)
from .diagnostic_render import render_four_state_event_window_diagnostic
from .schema import COMPONENT_COLUMNS, MAGNITUDE_COLUMNS, RESPONSE_COLUMNS, STATE_ORDER
from .summary import (
    FourStateEventWindowSummaryMatrix,
    four_state_event_window_summary_matrix,
    render_four_state_event_window_summary,
)

__all__ = [
    "COMPONENT_COLUMNS",
    "MAGNITUDE_COLUMNS",
    "RESPONSE_COLUMNS",
    "FourStateEventWindowDiagnostic",
    "FourStateEventWindowSummaryMatrix",
    "STATE_ORDER",
    "prepare_four_state_event_window_diagnostic",
    "render_four_state_event_window_diagnostic",
    "render_four_state_event_window_summary",
    "four_state_event_window_summary_matrix",
]
