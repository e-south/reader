"""Plate-reader response-window contracts and pure analysis primitives."""

from .contracts import (
    REQUEST_SCHEMA_VERSION,
    AggregationSpec,
    EventSpec,
    QualitySpec,
    ReductionSpec,
    ResponseSourceSpec,
    ResponseWindowRequest,
    load_response_window_request,
)
from .display import DISPLAY_SCHEMA_VERSION, DisplayExample, ResponseWindowDisplaySpec
from .reduction import TraceSummary, summarize_trace

__all__ = [
    "DISPLAY_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "AggregationSpec",
    "DisplayExample",
    "EventSpec",
    "QualitySpec",
    "ReductionSpec",
    "ResponseSourceSpec",
    "ResponseWindowDisplaySpec",
    "ResponseWindowRequest",
    "TraceSummary",
    "load_response_window_request",
    "summarize_trace",
]
