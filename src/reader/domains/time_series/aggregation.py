"""Aggregation policy kept separate from temporal trace reduction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

AggregationStatistic = Literal["mean", "median"]


@dataclass(frozen=True)
class ObservationAggregationSpec:
    """Separate within-unit observation reduction from the across-unit center."""

    within_unit_statistic: AggregationStatistic
    across_unit_statistic: AggregationStatistic

    def __post_init__(self) -> None:
        for field_name in ("within_unit_statistic", "across_unit_statistic"):
            if getattr(self, field_name) not in {"mean", "median"}:
                raise ValueError(f"observation_aggregation.{field_name} must be 'mean' or 'median'")

    @classmethod
    def from_mapping(cls, value: object) -> ObservationAggregationSpec:
        if not isinstance(value, Mapping):
            raise ValueError("observation_aggregation must be a mapping")
        payload = {str(key): item for key, item in value.items()}
        expected = {"within_unit_statistic", "across_unit_statistic"}
        unknown = sorted(set(payload) - expected)
        missing = sorted(expected - set(payload))
        if unknown:
            raise ValueError(f"observation_aggregation has unknown fields: {unknown}")
        if missing:
            raise ValueError(f"observation_aggregation is missing required fields: {missing}")
        return cls(
            within_unit_statistic=_statistic(
                payload["within_unit_statistic"],
                field="within_unit_statistic",
            ),
            across_unit_statistic=_statistic(
                payload["across_unit_statistic"],
                field="across_unit_statistic",
            ),
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "within_unit_statistic": self.within_unit_statistic,
            "across_unit_statistic": self.across_unit_statistic,
        }


def _statistic(value: object, *, field: str) -> AggregationStatistic:
    if value not in {"mean", "median"}:
        raise ValueError(f"observation_aggregation.{field} must be 'mean' or 'median'")
    return value  # type: ignore[return-value]


__all__ = ["AggregationStatistic", "ObservationAggregationSpec"]
