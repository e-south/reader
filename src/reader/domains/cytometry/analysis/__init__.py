"""Public event-table preparation and analysis for cytometry workflows."""

from reader.domains.cytometry.analysis.events import (
    CytometryAnalysis,
    CytometryAnalysisError,
    EventFilters,
    GateDefaults,
    GateSpec,
    NumericRange,
    ThresholdSpec,
    distinct_string_values,
    distinct_string_values_by_column,
    frame_columns,
    gate_defaults,
    prepare_event_preview,
    prepare_event_table,
    scan_event_table,
)
from reader.domains.cytometry.analysis.gating import analyze_events
from reader.domains.cytometry.analysis.plotting import prepare_plot_events, prepare_plot_payload

__all__ = [
    "CytometryAnalysis",
    "CytometryAnalysisError",
    "EventFilters",
    "GateDefaults",
    "GateSpec",
    "NumericRange",
    "ThresholdSpec",
    "analyze_events",
    "distinct_string_values",
    "distinct_string_values_by_column",
    "frame_columns",
    "gate_defaults",
    "prepare_event_preview",
    "prepare_event_table",
    "prepare_plot_events",
    "prepare_plot_payload",
    "scan_event_table",
]
