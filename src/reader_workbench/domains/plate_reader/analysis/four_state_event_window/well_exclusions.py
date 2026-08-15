"""Define explicit, reduction-bound exclusions for unsupported assay wells."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .contract_fields import exact_fields, nonempty

_COVERAGE_REASON = "insufficient_event_window_coverage"


@dataclass(frozen=True)
class WellExclusion:
    experiment_id: str
    design_id: str
    state: str
    position: str
    reduction_id: str
    reason: str


def parse_well_exclusions(value: object) -> tuple[WellExclusion, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("source.well_exclusions must be a sequence.")
    exclusions: list[WellExclusion] = []
    for index, item in enumerate(value):
        fields = {"experiment_id", "design_id", "state", "position", "reduction_id", "reason"}
        payload = exact_fields(item, context=f"source.well_exclusions[{index}]", required=fields)
        state = nonempty(payload["state"], context=f"source.well_exclusions[{index}].state")
        if state not in {"00", "10", "01", "11"}:
            raise ValueError(f"source.well_exclusions[{index}].state must be 00, 10, 01, or 11.")
        reason = nonempty(payload["reason"], context=f"source.well_exclusions[{index}].reason")
        if reason != _COVERAGE_REASON:
            raise ValueError(f"source.well_exclusions[{index}].reason must be {_COVERAGE_REASON!r}.")
        exclusions.append(
            WellExclusion(
                experiment_id=nonempty(
                    payload["experiment_id"], context=f"source.well_exclusions[{index}].experiment_id"
                ),
                design_id=nonempty(payload["design_id"], context=f"source.well_exclusions[{index}].design_id"),
                state=state,
                position=nonempty(payload["position"], context=f"source.well_exclusions[{index}].position"),
                reduction_id=nonempty(payload["reduction_id"], context=f"source.well_exclusions[{index}].reduction_id"),
                reason=reason,
            )
        )
    keys = [(item.experiment_id, item.position, item.reduction_id) for item in exclusions]
    if len(keys) != len(set(keys)):
        raise ValueError("source.well_exclusions must not repeat an experiment, position, and reduction.")
    return tuple(exclusions)


__all__ = ["WellExclusion", "parse_well_exclusions"]
