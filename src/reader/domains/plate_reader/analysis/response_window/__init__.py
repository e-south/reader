"""Plate-reader response-window contracts and pure analysis primitives."""

from .contracts import (
    AggregationSpec,
    EventSpec,
    QualitySpec,
    ReductionSpec,
    ResponseWindowAnalysisSpec,
    ResponseWindowSourceSpec,
)
from .display import DISPLAY_SCHEMA_VERSION, DisplayExample, ResponseWindowDisplaySpec
from .reduction import TraceSummary, summarize_trace

__all__ = [
    "DISPLAY_SCHEMA_VERSION",
    "AggregationSpec",
    "DisplayExample",
    "EventSpec",
    "QualitySpec",
    "ReductionSpec",
    "ResponseWindowAnalysisSpec",
    "ResponseWindowSourceSpec",
    "ResponseWindowDisplaySpec",
    "TraceSummary",
    "summarize_trace",
]
