"""Bind metric-neutral experiment states to the four-state SFXI contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SFXITreatmentSemantics:
    treatment_column: str
    corners: dict[str, str]
    case_sensitive: bool

    def inject(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of ``config`` with the resolved treatment contract."""

        resolved = dict(config)
        resolved["treatment_column"] = self.treatment_column
        resolved["treatment_map"] = dict(self.corners)
        resolved["treatment_case_sensitive"] = self.case_sensitive
        return resolved


def bind_sfxi_treatment_semantics(
    *,
    state_ids: Sequence[str],
    source_column: str,
    source_values: Mapping[str, str],
    case_sensitive: bool,
    treatment_column: str | None = None,
) -> SFXITreatmentSemantics:
    """Validate explicit state-space values for the four-state SFXI transform."""

    normalized_state_ids = tuple(str(state_id) for state_id in state_ids)
    if normalized_state_ids != ("00", "10", "01", "11"):
        raise ValueError("SFXI state space must declare exactly 00, 10, 01, 11 in that order")
    if set(source_values) != set(normalized_state_ids):
        raise ValueError("SFXI state values must define exactly 00, 10, 01, and 11")
    column = treatment_column or source_column
    if not isinstance(column, str) or not column.strip():
        raise ValueError("sfxi treatment column must be a non-empty string")
    return SFXITreatmentSemantics(
        treatment_column=column.strip(),
        corners={state_id: str(source_values[state_id]) for state_id in normalized_state_ids},
        case_sensitive=bool(case_sensitive),
    )


__all__ = ["SFXITreatmentSemantics", "bind_sfxi_treatment_semantics"]
