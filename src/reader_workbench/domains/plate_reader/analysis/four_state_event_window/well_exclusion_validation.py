"""Verify that declared well exclusions correspond to missing temporal support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from .contracts import QualitySpec, ReductionSpec
from .reduction import four_state_event_window_temporal_spec, reduce_temporal_trace
from .well_exclusions import WellExclusion


def require_well_exclusion_support_failure(
    exclusions: Sequence[WellExclusion],
    *,
    reductions: Mapping[str, ReductionSpec],
    quality: QualitySpec,
    response: pd.DataFrame,
    magnitude: pd.DataFrame,
    event_estimates_h: Sequence[float],
    experiment_id: str,
) -> None:
    for exclusion in exclusions:
        try:
            reduction = reductions[exclusion.reduction_id]
        except KeyError as exc:
            raise ValueError(
                f"{experiment_id}: well exclusion names unknown reduction {exclusion.reduction_id!r}."
            ) from exc
        support_failures: list[str] = []
        for frame in (response, magnitude):
            trace = frame.loc[frame["position"].astype(str).eq(exclusion.position)]
            for event_estimate_h in event_estimates_h:
                try:
                    reduce_temporal_trace(
                        trace["time"].to_numpy(dtype=float),
                        trace["value"].to_numpy(dtype=float),
                        spec=four_state_event_window_temporal_spec(reduction, quality),
                        trace_id=f"{experiment_id}:{exclusion.position}:{reduction.id}:support",
                        origin_h=event_estimate_h,
                        policy_clipped=trace["value_policy_clipped"].to_numpy(dtype=bool),
                        instrument_overflow=trace["value_instrument_overflow"].to_numpy(dtype=bool),
                        bound_kinds=trace["value_bound_kind"].to_numpy(dtype=object),
                    )
                except ValueError as exc:
                    if _is_support_failure(str(exc)):
                        support_failures.append(str(exc))
        if not support_failures:
            raise ValueError(
                f"{experiment_id}:{exclusion.position}:{exclusion.reduction_id} "
                "does not have insufficient event-window coverage."
            )


def _is_support_failure(message: str) -> bool:
    return any(
        token in message
        for token in (
            "does not cover",
            "interior gap",
            "observations in the selected interval",
            "requires observed interval boundaries",
            "selected interval contains no observations",
        )
    )


__all__ = ["require_well_exclusion_support_failure"]
