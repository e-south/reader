"""Plate-reader four-state event-window contracts and pure analysis primitives."""

from .contracts import (
    AggregationSpec,
    EventSpec,
    FourStateEventWindowAnalysisSpec,
    FourStateEventWindowSourceSpec,
    QualitySpec,
    ReductionSpec,
)
from .reduction import four_state_event_window_temporal_spec

__all__ = [
    "AggregationSpec",
    "EventSpec",
    "QualitySpec",
    "ReductionSpec",
    "FourStateEventWindowAnalysisSpec",
    "FourStateEventWindowSourceSpec",
    "four_state_event_window_temporal_spec",
]
