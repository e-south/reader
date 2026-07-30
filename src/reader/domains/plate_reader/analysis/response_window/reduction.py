"""Response-window adapter to the assay-neutral temporal reducer."""

from __future__ import annotations

from reader.domains.time_series import (
    IntervalSelection,
    TemporalReductionSpec,
    TemporalSupportPolicy,
    combine_bound_kinds,
    invert_bound_kind,
    reduce_temporal_trace,
)

from .contracts import QualitySpec, ReductionSpec


def response_window_temporal_spec(
    reduction: ReductionSpec,
    quality: QualitySpec,
    *,
    absolute_window_h: tuple[float, float] | None = None,
) -> TemporalReductionSpec:
    """Project response-window policy into the neutral temporal contract.

    Response-window reductions remain event-relative unless an explicit
    absolute pre-event interval is supplied. The projection preserves the
    established interpolation, log2 output, support-gap, and censor behavior.
    """

    if absolute_window_h is None:
        selection = IntervalSelection(
            time_basis="event_relative",
            start_h=reduction.window_start_event_h,
            end_h=reduction.window_end_event_h,
        )
    else:
        selection = IntervalSelection(
            time_basis="absolute",
            start_h=float(absolute_window_h[0]),
            end_h=float(absolute_window_h[1]),
        )
    return TemporalReductionSpec(
        selection=selection,
        method=reduction.method,
        output_space="log2",
        support=TemporalSupportPolicy(
            boundary_support="covered",
            minimum_observations=0,
            maximum_interior_gap_h=quality.max_interior_gap_h,
            positive_floor=quality.positive_floor,
            positive_value_scope="entire_trace",
            censored_values="allow",
        ),
    )


__all__ = [
    "combine_bound_kinds",
    "invert_bound_kind",
    "reduce_temporal_trace",
    "response_window_temporal_spec",
]
