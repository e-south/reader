"""Cross-record verification for clipping and censor-bound provenance."""

from __future__ import annotations

import pandas as pd

from reader.domains.plate_reader.analysis.response_window.contracts import ResponseWindowRequest
from reader.domains.plate_reader.analysis.response_window.reduction import combine_bound_kinds, invert_bound_kind
from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER

from .verification_trace_support import verify_trace_support


def verify_value_provenance(
    designs: pd.DataFrame,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    *,
    events: pd.DataFrame,
    request: ResponseWindowRequest,
) -> None:
    trace_affected = traces["value_policy_clipped"] | traces["value_instrument_overflow"]
    if not trace_affected.eq(traces["value_bound_kind"].ne("exact")).all():
        raise ValueError("response-window trace value provenance disagrees with its bound kind.")
    for signal in ("response", "magnitude"):
        affected = wells[f"{signal}_policy_clipped_point_count"].gt(0) | wells[
            f"{signal}_instrument_overflow_point_count"
        ].gt(0)
        if not affected.eq(wells[f"{signal}_bound_kind"].ne("exact")).all():
            raise ValueError("response-window well value provenance disagrees with its bound kind.")

    verify_trace_support(designs, wells, traces, events=events, request=request)

    for row in designs.itertuples(index=False):
        for state in STATE_ORDER:
            identity = {"experiment_id": str(row.experiment_id), "reduction_id": str(row.reduction_id), "state": state}
            design_wells = _state_wells(wells, design_id=str(row.design_id), **identity)
            reference_wells = _state_wells(wells, design_id=str(row.reference_design_id), **identity)
            expected = _expected_state_provenance(
                design_wells,
                reference_wells,
                is_reference=str(row.design_id) == str(row.reference_design_id),
            )
            observed = tuple(
                getattr(row, field)
                for field in (
                    f"r{state}_has_policy_clipping",
                    f"r{state}_has_instrument_overflow",
                    f"r{state}_bound_kind",
                    f"b{state}_has_policy_clipping",
                    f"b{state}_has_instrument_overflow",
                    f"b{state}_bound_kind",
                )
            )
            if observed != expected:
                raise ValueError("response-window design state provenance disagrees with well support.")


def _state_wells(
    wells: pd.DataFrame,
    *,
    experiment_id: str,
    reduction_id: str,
    design_id: str,
    state: str,
) -> pd.DataFrame:
    return wells.loc[
        wells["experiment_id"].astype(str).eq(experiment_id)
        & wells["design_id"].astype(str).eq(design_id)
        & wells["reduction_id"].astype(str).eq(reduction_id)
        & wells["state"].astype(str).eq(state)
    ]


def _expected_state_provenance(
    design_wells: pd.DataFrame,
    reference_wells: pd.DataFrame,
    *,
    is_reference: bool,
) -> tuple[object, ...]:
    response_policy = bool(design_wells["response_policy_clipped_point_count"].gt(0).any())
    response_overflow = bool(design_wells["response_instrument_overflow_point_count"].gt(0).any())
    magnitude_policy = bool(
        design_wells["magnitude_policy_clipped_point_count"].gt(0).any()
        or reference_wells["magnitude_policy_clipped_point_count"].gt(0).any()
    )
    magnitude_overflow = bool(
        design_wells["magnitude_instrument_overflow_point_count"].gt(0).any()
        or reference_wells["magnitude_instrument_overflow_point_count"].gt(0).any()
    )
    return (
        response_policy,
        response_overflow,
        combine_bound_kinds(*design_wells["response_bound_kind"]),
        magnitude_policy,
        magnitude_overflow,
        (
            "exact"
            if is_reference
            else combine_bound_kinds(
                combine_bound_kinds(*design_wells["magnitude_bound_kind"]),
                invert_bound_kind(combine_bound_kinds(*reference_wells["magnitude_bound_kind"])),
            )
        ),
    )


__all__ = ["verify_value_provenance"]
