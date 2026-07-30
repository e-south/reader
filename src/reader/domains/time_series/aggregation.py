"""Aggregation policy kept separate from temporal trace reduction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

AggregationStatistic = Literal["mean", "median"]


@dataclass(frozen=True)
class ReplicateAggregationSpec:
    """Distinguish technical collapse from the across-replicate center."""

    technical_replicate_statistic: AggregationStatistic
    replicate_center_statistic: AggregationStatistic

    def __post_init__(self) -> None:
        for field_name in ("technical_replicate_statistic", "replicate_center_statistic"):
            if getattr(self, field_name) not in {"mean", "median"}:
                raise ValueError(f"replicate_aggregation.{field_name} must be 'mean' or 'median'")

    @classmethod
    def from_mapping(cls, value: object) -> ReplicateAggregationSpec:
        if not isinstance(value, Mapping):
            raise ValueError("replicate_aggregation must be a mapping")
        payload = {str(key): item for key, item in value.items()}
        expected = {"technical_replicate_statistic", "replicate_center_statistic"}
        unknown = sorted(set(payload) - expected)
        missing = sorted(expected - set(payload))
        if unknown:
            raise ValueError(f"replicate_aggregation has unknown fields: {unknown}")
        if missing:
            raise ValueError(f"replicate_aggregation is missing required fields: {missing}")
        return cls(
            technical_replicate_statistic=_statistic(
                payload["technical_replicate_statistic"],
                field="technical_replicate_statistic",
            ),
            replicate_center_statistic=_statistic(
                payload["replicate_center_statistic"],
                field="replicate_center_statistic",
            ),
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "technical_replicate_statistic": self.technical_replicate_statistic,
            "replicate_center_statistic": self.replicate_center_statistic,
        }


def _statistic(value: object, *, field: str) -> AggregationStatistic:
    if value not in {"mean", "median"}:
        raise ValueError(f"replicate_aggregation.{field} must be 'mean' or 'median'")
    return value  # type: ignore[return-value]


__all__ = ["AggregationStatistic", "ReplicateAggregationSpec"]
