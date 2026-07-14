"""Resolve SFXI treatment states from an experiment logic-map contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SFXITreatmentSemantics:
    logic_map_ref: str
    treatment_column: str
    corners: dict[str, str]
    case_sensitive: bool

    def inject(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``config`` with the resolved treatment contract."""

        resolved = dict(config)
        resolved["logic_map_ref"] = self.logic_map_ref
        resolved["treatment_column"] = self.treatment_column
        resolved["treatment_map"] = dict(self.corners)
        resolved["treatment_case_sensitive"] = self.case_sensitive
        return resolved


def resolve_sfxi_treatment_semantics(
    *,
    ctx: Any,
    logic_map_ref: str,
    treatment_column: str | None = None,
) -> SFXITreatmentSemantics:
    """Resolve one exact 00/10/01/11 mapping from Reader experiment semantics."""

    if ctx.experiment is None:
        raise ValueError("sfxi requires experiment semantics in the run context")
    ref = str(logic_map_ref).strip()
    if not ref:
        raise ValueError("sfxi.logic_map_ref must be a non-empty string")
    logic_map = ctx.experiment.annotations.resolve_logic_map(ref=ref)
    column = treatment_column or logic_map.column
    if not isinstance(column, str) or not column.strip():
        raise ValueError("sfxi treatment column must be a non-empty string")
    return SFXITreatmentSemantics(
        logic_map_ref=ref,
        treatment_column=column.strip(),
        corners=dict(logic_map.corners),
        case_sensitive=bool(logic_map.case_sensitive),
    )


__all__ = ["SFXITreatmentSemantics", "resolve_sfxi_treatment_semantics"]
