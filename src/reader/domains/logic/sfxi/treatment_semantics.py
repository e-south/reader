"""Bind metric-neutral experiment states to the four-state SFXI contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SFXITreatmentSemantics:
    state_map_ref: str
    treatment_column: str
    corners: dict[str, str]
    case_sensitive: bool

    def inject(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``config`` with the resolved treatment contract."""

        resolved = dict(config)
        resolved["state_map_ref"] = self.state_map_ref
        resolved["treatment_column"] = self.treatment_column
        resolved["treatment_map"] = dict(self.corners)
        resolved["treatment_case_sensitive"] = self.case_sensitive
        return resolved


def resolve_sfxi_treatment_semantics(
    *,
    ctx: Any,
    state_map_ref: str,
    treatment_column: str | None = None,
) -> SFXITreatmentSemantics:
    """Resolve one exact 00/10/01/11 mapping from Reader experiment semantics."""

    if ctx.experiment is None:
        raise ValueError("sfxi requires experiment semantics in the run context")
    ref = str(state_map_ref).strip()
    if not ref:
        raise ValueError("sfxi.state_map_ref must be a non-empty string")
    state_space = ctx.experiment.annotations.resolve_ordered_state_space(ref=ref)
    if state_space.state_ids != ("00", "10", "01", "11"):
        raise ValueError("SFXI state space must declare exactly 00, 10, 01, 11 in that order")
    column = treatment_column or state_space.column
    if not isinstance(column, str) or not column.strip():
        raise ValueError("sfxi treatment column must be a non-empty string")
    return SFXITreatmentSemantics(
        state_map_ref=ref,
        treatment_column=column.strip(),
        corners=dict(state_space.source_values),
        case_sensitive=bool(state_space.case_sensitive),
    )


__all__ = ["SFXITreatmentSemantics", "resolve_sfxi_treatment_semantics"]
