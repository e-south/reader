"""Strict contracts for event-relative plate-reader response summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ReductionMethod = Literal["geometric_time_mean", "integrated_linear_mean"]
ResponseBasis = Literal["post_window", "post_minus_pre"]
ReductionRole = Literal["primary", "sensitivity"]
ReplicateStat = Literal["mean", "median"]
EventEstimateMethod = Literal["segment_gap_midpoint"]


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _exact_fields(
    value: object,
    *,
    context: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    payload = _mapping(value, context=context)
    allowed = required | (optional or set())
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {unknown}.")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}.")
    return payload


def _nonempty(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be a finite number.")
    return result


@dataclass(frozen=True)
class ResponseWindowSourceSpec:
    response_channel: str
    magnitude_channel: str
    growth_channel: str
    reference_design_id: str
    state_column: str
    state_values: Mapping[str, str]
    state_values_case_sensitive: bool = True

    @classmethod
    def from_mapping(cls, value: object) -> ResponseWindowSourceSpec:
        fields = {
            "response_channel",
            "magnitude_channel",
            "growth_channel",
            "reference_design_id",
            "state_column",
            "state_values",
        }
        payload = _exact_fields(
            value,
            context="source",
            required=fields,
            optional={"state_values_case_sensitive"},
        )
        state_values = _mapping(payload["state_values"], context="source.state_values")
        expected_states = {"00", "10", "01", "11"}
        if set(state_values) != expected_states:
            raise ValueError("source.state_values must define exactly 00, 10, 01, and 11.")
        normalized_values = {
            state: _nonempty(state_values[state], context=f"source.state_values.{state}")
            for state in sorted(expected_states)
        }
        if len(set(normalized_values.values())) != 4:
            raise ValueError("source.state_values must map to four distinct source values.")
        case_sensitive = payload.get("state_values_case_sensitive", True)
        if not isinstance(case_sensitive, bool):
            raise ValueError("source.state_values_case_sensitive must be true or false.")
        return cls(
            response_channel=_nonempty(payload["response_channel"], context="source.response_channel"),
            magnitude_channel=_nonempty(payload["magnitude_channel"], context="source.magnitude_channel"),
            growth_channel=_nonempty(payload["growth_channel"], context="source.growth_channel"),
            reference_design_id=_nonempty(payload["reference_design_id"], context="source.reference_design_id"),
            state_column=_nonempty(payload["state_column"], context="source.state_column"),
            state_values=normalized_values,
            state_values_case_sensitive=case_sensitive,
        )


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    event_kind: str
    segment_column: str
    pre_segment_index: int
    post_segment_index: int
    estimate_method: EventEstimateMethod
    declaration: str

    @classmethod
    def from_mapping(cls, value: object) -> EventSpec:
        fields = {
            "event_id",
            "event_kind",
            "segment_column",
            "pre_segment_index",
            "post_segment_index",
            "estimate_method",
            "declaration",
        }
        payload = _exact_fields(value, context="event", required=fields)
        pre = payload["pre_segment_index"]
        post = payload["post_segment_index"]
        if isinstance(pre, bool) or not isinstance(pre, int):
            raise ValueError("event.pre_segment_index must be an integer.")
        if isinstance(post, bool) or not isinstance(post, int):
            raise ValueError("event.post_segment_index must be an integer.")
        if pre == post:
            raise ValueError("event segment indexes must differ.")
        estimate = _nonempty(payload["estimate_method"], context="event.estimate_method")
        if estimate != "segment_gap_midpoint":
            raise ValueError("event.estimate_method must be 'segment_gap_midpoint' for symmetric event uncertainty.")
        return cls(
            event_id=_nonempty(payload["event_id"], context="event.event_id"),
            event_kind=_nonempty(payload["event_kind"], context="event.event_kind"),
            segment_column=_nonempty(payload["segment_column"], context="event.segment_column"),
            pre_segment_index=pre,
            post_segment_index=post,
            estimate_method=estimate,
            declaration=_nonempty(payload["declaration"], context="event.declaration"),
        )


@dataclass(frozen=True)
class ReductionSpec:
    id: str
    window_start_event_h: float
    window_end_event_h: float
    method: ReductionMethod
    response_basis: ResponseBasis
    role: ReductionRole
    pre_window_duration_h: float | None = None

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> ReductionSpec:
        required = {
            "id",
            "window_start_event_h",
            "window_end_event_h",
            "method",
            "response_basis",
            "role",
        }
        payload = _exact_fields(
            value,
            context=f"reductions[{index}]",
            required=required,
            optional={"pre_window_duration_h"},
        )
        start = _finite(payload["window_start_event_h"], context=f"reductions[{index}].window_start_event_h")
        end = _finite(payload["window_end_event_h"], context=f"reductions[{index}].window_end_event_h")
        if start < 0.0 or end <= start:
            raise ValueError(f"reductions[{index}] requires 0 <= window start < window end.")
        method = _nonempty(payload["method"], context=f"reductions[{index}].method")
        if method not in {"geometric_time_mean", "integrated_linear_mean"}:
            raise ValueError(f"reductions[{index}].method is unsupported: {method!r}.")
        basis = _nonempty(payload["response_basis"], context=f"reductions[{index}].response_basis")
        if basis not in {"post_window", "post_minus_pre"}:
            raise ValueError(f"reductions[{index}].response_basis is unsupported: {basis!r}.")
        role = _nonempty(payload["role"], context=f"reductions[{index}].role")
        if role not in {"primary", "sensitivity"}:
            raise ValueError(f"reductions[{index}].role is unsupported: {role!r}.")
        raw_pre = payload.get("pre_window_duration_h")
        pre = None if raw_pre is None else _finite(raw_pre, context=f"reductions[{index}].pre_window_duration_h")
        if basis == "post_minus_pre" and (pre is None or pre <= 0.0):
            raise ValueError(f"reductions[{index}] post_minus_pre requires a positive pre_window_duration_h.")
        if basis == "post_window" and pre is not None:
            raise ValueError(f"reductions[{index}] pre_window_duration_h is only valid for post_minus_pre.")
        return cls(
            id=_nonempty(payload["id"], context=f"reductions[{index}].id"),
            window_start_event_h=start,
            window_end_event_h=end,
            method=method,
            response_basis=basis,
            role=role,
            pre_window_duration_h=pre,
        )


@dataclass(frozen=True)
class AggregationSpec:
    replicate_stat: ReplicateStat
    bootstrap_samples: int
    confidence_level: float
    random_seed: int

    @classmethod
    def from_mapping(cls, value: object) -> AggregationSpec:
        fields = {"replicate_stat", "bootstrap_samples", "confidence_level", "random_seed"}
        payload = _exact_fields(value, context="aggregation", required=fields)
        stat = _nonempty(payload["replicate_stat"], context="aggregation.replicate_stat")
        if stat not in {"mean", "median"}:
            raise ValueError(f"aggregation.replicate_stat is unsupported: {stat!r}.")
        samples = payload["bootstrap_samples"]
        seed = payload["random_seed"]
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
            raise ValueError("aggregation.bootstrap_samples must be an integer of at least 100.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("aggregation.random_seed must be a non-negative integer.")
        level = _finite(payload["confidence_level"], context="aggregation.confidence_level")
        if not 0.5 < level < 1.0:
            raise ValueError("aggregation.confidence_level must be between 0.5 and 1.0.")
        return cls(replicate_stat=stat, bootstrap_samples=samples, confidence_level=level, random_seed=seed)


@dataclass(frozen=True)
class QualitySpec:
    positive_floor: float
    max_interior_gap_h: float
    min_replicates_per_state: int

    @classmethod
    def from_mapping(cls, value: object) -> QualitySpec:
        fields = {"positive_floor", "max_interior_gap_h", "min_replicates_per_state"}
        payload = _exact_fields(value, context="quality", required=fields)
        floor = _finite(payload["positive_floor"], context="quality.positive_floor")
        gap = _finite(payload["max_interior_gap_h"], context="quality.max_interior_gap_h")
        replicates = payload["min_replicates_per_state"]
        if floor <= 0.0:
            raise ValueError("quality.positive_floor must be positive.")
        if gap <= 0.0:
            raise ValueError("quality.max_interior_gap_h must be positive.")
        if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
            raise ValueError("quality.min_replicates_per_state must be an integer of at least 2.")
        return cls(positive_floor=floor, max_interior_gap_h=gap, min_replicates_per_state=replicates)


@dataclass(frozen=True)
class ResponseWindowAnalysisSpec:
    source: ResponseWindowSourceSpec
    event: EventSpec
    reductions: tuple[ReductionSpec, ...]
    aggregation: AggregationSpec
    quality: QualitySpec

    @property
    def primary_reduction(self) -> ReductionSpec:
        return next(spec for spec in self.reductions if spec.role == "primary")

    @classmethod
    def from_mapping(cls, value: object) -> ResponseWindowAnalysisSpec:
        fields = {
            "source",
            "event",
            "reductions",
            "aggregation",
            "quality",
        }
        payload = _exact_fields(value, context="analysis", required=fields)
        raw_reductions = payload["reductions"]
        if isinstance(raw_reductions, (str, bytes)) or not isinstance(raw_reductions, Sequence):
            raise ValueError("reductions must be a non-empty sequence.")
        reductions = tuple(ReductionSpec.from_mapping(item, index=index) for index, item in enumerate(raw_reductions))
        ids = [spec.id for spec in reductions]
        if not reductions or len(ids) != len(set(ids)):
            raise ValueError("reduction ids must be non-empty and unique.")
        if sum(spec.role == "primary" for spec in reductions) != 1:
            raise ValueError("analysis must declare exactly one primary reduction.")
        return cls(
            source=ResponseWindowSourceSpec.from_mapping(payload["source"]),
            event=EventSpec.from_mapping(payload["event"]),
            reductions=reductions,
            aggregation=AggregationSpec.from_mapping(payload["aggregation"]),
            quality=QualitySpec.from_mapping(payload["quality"]),
        )


__all__ = [
    "AggregationSpec",
    "EventSpec",
    "QualitySpec",
    "ReductionSpec",
    "ResponseWindowAnalysisSpec",
    "ResponseWindowSourceSpec",
]
