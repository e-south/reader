"""Plate-reader response-window contracts and pure analysis primitives."""

from .contracts import (
    AggregationSpec,
    EventSpec,
    QualitySpec,
    ReductionSpec,
    ResponseWindowAnalysisSpec,
    ResponseWindowSourceSpec,
)
from .reduction import TraceSummary, summarize_trace

__all__ = [
    "AggregationSpec",
    "EventSpec",
    "QualitySpec",
    "ReductionSpec",
    "ResponseWindowAnalysisSpec",
    "ResponseWindowSourceSpec",
    "TraceSummary",
    "summarize_trace",
]
