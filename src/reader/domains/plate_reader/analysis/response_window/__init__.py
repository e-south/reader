"""Plate-reader response-window contracts and pure analysis primitives."""

from .contracts import (
    AggregationSpec,
    EventSpec,
    QualitySpec,
    ReductionSpec,
    ResponseWindowAnalysisSpec,
    ResponseWindowSourceSpec,
)
from .reduction import response_window_temporal_spec

__all__ = [
    "AggregationSpec",
    "EventSpec",
    "QualitySpec",
    "ReductionSpec",
    "ResponseWindowAnalysisSpec",
    "ResponseWindowSourceSpec",
    "response_window_temporal_spec",
]
