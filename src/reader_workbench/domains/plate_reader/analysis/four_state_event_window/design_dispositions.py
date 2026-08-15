"""Define plate- and reduction-scoped design eligibility dispositions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .contract_fields import exact_fields, nonempty

_OBSERVATION_REASON = "insufficient_state_observations"


@dataclass(frozen=True)
class DesignDisposition:
    experiment_id: str
    design_id: str
    state: str
    reduction_id: str
    reason: str


def parse_design_dispositions(value: object) -> tuple[DesignDisposition, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("source.design_dispositions must be a sequence.")
    dispositions: list[DesignDisposition] = []
    for index, item in enumerate(value):
        context = f"source.design_dispositions[{index}]"
        payload = exact_fields(
            item,
            context=context,
            required={"experiment_id", "design_id", "state", "reduction_id", "reason"},
        )
        state = nonempty(payload["state"], context=f"{context}.state")
        if state not in {"00", "10", "01", "11"}:
            raise ValueError(f"{context}.state must be 00, 10, 01, or 11.")
        reason = nonempty(payload["reason"], context=f"{context}.reason")
        if reason != _OBSERVATION_REASON:
            raise ValueError(f"{context}.reason must be {_OBSERVATION_REASON!r}.")
        dispositions.append(
            DesignDisposition(
                experiment_id=nonempty(payload["experiment_id"], context=f"{context}.experiment_id"),
                design_id=nonempty(payload["design_id"], context=f"{context}.design_id"),
                state=state,
                reduction_id=nonempty(payload["reduction_id"], context=f"{context}.reduction_id"),
                reason=reason,
            )
        )
    keys = [(item.experiment_id, item.design_id, item.reduction_id) for item in dispositions]
    if len(keys) != len(set(keys)):
        raise ValueError("source.design_dispositions must not repeat an experiment, design, and reduction.")
    return tuple(dispositions)


__all__ = ["DesignDisposition", "parse_design_dispositions"]
