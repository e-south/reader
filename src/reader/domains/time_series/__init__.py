"""Assay-neutral temporal selection and reduction primitives."""

from .aggregation import ObservationAggregationSpec
from .contracts import (
    EndpointSelection,
    IntervalSelection,
    TemporalReductionSpec,
    TemporalSupportPolicy,
    parse_temporal_selection,
)
from .reduction import (
    TemporalReductionResult,
    combine_bound_kinds,
    invert_bound_kind,
    reduce_temporal_trace,
)

__all__ = [
    "EndpointSelection",
    "IntervalSelection",
    "ObservationAggregationSpec",
    "TemporalReductionResult",
    "TemporalReductionSpec",
    "TemporalSupportPolicy",
    "combine_bound_kinds",
    "invert_bound_kind",
    "parse_temporal_selection",
    "reduce_temporal_trace",
]
