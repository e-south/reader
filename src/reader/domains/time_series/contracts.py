"""Strict contracts for assay-neutral temporal reductions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

TimeBasis = Literal["absolute", "event_relative"]
EndpointMode = Literal["exact", "nearest"]
IntervalBoundary = Literal["inclusive"]
TemporalReductionMethod = Literal[
    "identity",
    "observed_mean",
    "observed_median",
    "geometric_time_mean",
    "integrated_linear_mean",
]
OutputSpace = Literal["linear", "log2"]
BoundarySupport = Literal["none", "covered", "observed"]
CensoredValuePolicy = Literal["allow", "reject"]
PositiveValueScope = Literal["selected_support", "entire_trace"]


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
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
        raise ValueError(f"{context} has unknown fields: {unknown}")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")
    return payload


def _choice(value: object, *, context: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{context} must be one of: {options}")
    return value


def _finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


@dataclass(frozen=True)
class EndpointSelection:
    """One recorded or event-relative endpoint."""

    time_basis: TimeBasis
    time_h: float
    mode: EndpointMode
    tolerance_h: float

    def __post_init__(self) -> None:
        if self.time_basis not in {"absolute", "event_relative"}:
            raise ValueError("endpoint.time_basis must be 'absolute' or 'event_relative'")
        if self.mode not in {"exact", "nearest"}:
            raise ValueError("endpoint.mode must be 'exact' or 'nearest'")
        if not math.isfinite(float(self.time_h)):
            raise ValueError("endpoint.time_h must be finite")
        if not math.isfinite(float(self.tolerance_h)) or float(self.tolerance_h) < 0.0:
            raise ValueError("endpoint.tolerance_h must be finite and non-negative")
        if self.mode == "exact" and float(self.tolerance_h) != 0.0:
            raise ValueError("endpoint.tolerance_h must be zero when mode is 'exact'")

    @classmethod
    def from_mapping(cls, value: object) -> EndpointSelection:
        payload = _exact_fields(
            value,
            context="temporal_reduction.selection",
            required={"kind", "time_basis", "time_h", "mode", "tolerance_h"},
        )
        if payload["kind"] != "endpoint":
            raise ValueError("temporal_reduction.selection.kind must be 'endpoint'")
        return cls(
            time_basis=_choice(
                payload["time_basis"],
                context="temporal_reduction.selection.time_basis",
                allowed={"absolute", "event_relative"},
            ),  # type: ignore[arg-type]
            time_h=_finite(payload["time_h"], context="temporal_reduction.selection.time_h"),
            mode=_choice(
                payload["mode"],
                context="temporal_reduction.selection.mode",
                allowed={"exact", "nearest"},
            ),  # type: ignore[arg-type]
            tolerance_h=_finite(
                payload["tolerance_h"],
                context="temporal_reduction.selection.tolerance_h",
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": "endpoint",
            "time_basis": self.time_basis,
            "time_h": self.time_h,
            "mode": self.mode,
            "tolerance_h": self.tolerance_h,
        }


@dataclass(frozen=True)
class IntervalSelection:
    """One closed recorded or event-relative interval."""

    time_basis: TimeBasis
    start_h: float
    end_h: float
    boundary: IntervalBoundary = "inclusive"

    def __post_init__(self) -> None:
        if self.time_basis not in {"absolute", "event_relative"}:
            raise ValueError("interval.time_basis must be 'absolute' or 'event_relative'")
        if self.boundary != "inclusive":
            raise ValueError("interval.boundary must be 'inclusive'")
        if not math.isfinite(float(self.start_h)) or not math.isfinite(float(self.end_h)):
            raise ValueError("interval bounds must be finite")
        if float(self.end_h) <= float(self.start_h):
            raise ValueError("interval requires start_h < end_h")

    @classmethod
    def from_mapping(cls, value: object) -> IntervalSelection:
        payload = _exact_fields(
            value,
            context="temporal_reduction.selection",
            required={"kind", "time_basis", "start_h", "end_h", "boundary"},
        )
        if payload["kind"] != "interval":
            raise ValueError("temporal_reduction.selection.kind must be 'interval'")
        return cls(
            time_basis=_choice(
                payload["time_basis"],
                context="temporal_reduction.selection.time_basis",
                allowed={"absolute", "event_relative"},
            ),  # type: ignore[arg-type]
            start_h=_finite(payload["start_h"], context="temporal_reduction.selection.start_h"),
            end_h=_finite(payload["end_h"], context="temporal_reduction.selection.end_h"),
            boundary=_choice(
                payload["boundary"],
                context="temporal_reduction.selection.boundary",
                allowed={"inclusive"},
            ),  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": "interval",
            "time_basis": self.time_basis,
            "start_h": self.start_h,
            "end_h": self.end_h,
            "boundary": self.boundary,
        }


type TemporalSelection = EndpointSelection | IntervalSelection


def parse_temporal_selection(value: object) -> TemporalSelection:
    payload = _mapping(value, context="temporal_reduction.selection")
    kind = payload.get("kind")
    if kind == "endpoint":
        return EndpointSelection.from_mapping(payload)
    if kind == "interval":
        return IntervalSelection.from_mapping(payload)
    raise ValueError("temporal_reduction.selection.kind must be 'endpoint' or 'interval'")


@dataclass(frozen=True)
class TemporalSupportPolicy:
    """Trace support and value-provenance requirements."""

    boundary_support: BoundarySupport
    minimum_observations: int
    maximum_interior_gap_h: float | None
    positive_floor: float | None
    positive_value_scope: PositiveValueScope
    censored_values: CensoredValuePolicy

    def __post_init__(self) -> None:
        if self.boundary_support not in {"none", "covered", "observed"}:
            raise ValueError("support.boundary_support must be 'none', 'covered', or 'observed'")
        if (
            isinstance(self.minimum_observations, bool)
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations < 0
        ):
            raise ValueError("support.minimum_observations must be a non-negative integer")
        if self.maximum_interior_gap_h is not None and (
            not math.isfinite(float(self.maximum_interior_gap_h)) or float(self.maximum_interior_gap_h) <= 0.0
        ):
            raise ValueError("support.maximum_interior_gap_h must be null or a positive finite number")
        if self.positive_floor is not None and (
            not math.isfinite(float(self.positive_floor)) or float(self.positive_floor) <= 0.0
        ):
            raise ValueError("support.positive_floor must be null or a positive finite number")
        if self.positive_value_scope not in {"selected_support", "entire_trace"}:
            raise ValueError("support.positive_value_scope must be 'selected_support' or 'entire_trace'")
        if self.censored_values not in {"allow", "reject"}:
            raise ValueError("support.censored_values must be 'allow' or 'reject'")

    @classmethod
    def from_mapping(cls, value: object) -> TemporalSupportPolicy:
        payload = _exact_fields(
            value,
            context="temporal_reduction.support",
            required={
                "boundary_support",
                "minimum_observations",
                "maximum_interior_gap_h",
                "positive_floor",
                "positive_value_scope",
                "censored_values",
            },
        )
        minimum = payload["minimum_observations"]
        if isinstance(minimum, bool) or not isinstance(minimum, int):
            raise ValueError("temporal_reduction.support.minimum_observations must be an integer")
        maximum_gap = payload["maximum_interior_gap_h"]
        positive_floor = payload["positive_floor"]
        return cls(
            boundary_support=_choice(
                payload["boundary_support"],
                context="temporal_reduction.support.boundary_support",
                allowed={"none", "covered", "observed"},
            ),  # type: ignore[arg-type]
            minimum_observations=minimum,
            maximum_interior_gap_h=(
                None
                if maximum_gap is None
                else _finite(maximum_gap, context="temporal_reduction.support.maximum_interior_gap_h")
            ),
            positive_floor=(
                None
                if positive_floor is None
                else _finite(positive_floor, context="temporal_reduction.support.positive_floor")
            ),
            positive_value_scope=_choice(
                payload["positive_value_scope"],
                context="temporal_reduction.support.positive_value_scope",
                allowed={"selected_support", "entire_trace"},
            ),  # type: ignore[arg-type]
            censored_values=_choice(
                payload["censored_values"],
                context="temporal_reduction.support.censored_values",
                allowed={"allow", "reject"},
            ),  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "boundary_support": self.boundary_support,
            "minimum_observations": self.minimum_observations,
            "maximum_interior_gap_h": self.maximum_interior_gap_h,
            "positive_floor": self.positive_floor,
            "positive_value_scope": self.positive_value_scope,
            "censored_values": self.censored_values,
        }


@dataclass(frozen=True)
class TemporalReductionSpec:
    """One selection, numerical method, value space, and support policy."""

    selection: TemporalSelection
    method: TemporalReductionMethod
    output_space: OutputSpace
    support: TemporalSupportPolicy

    def __post_init__(self) -> None:
        methods = {
            "identity",
            "observed_mean",
            "observed_median",
            "geometric_time_mean",
            "integrated_linear_mean",
        }
        if self.method not in methods:
            raise ValueError(f"temporal_reduction.method is unsupported: {self.method!r}")
        if self.output_space not in {"linear", "log2"}:
            raise ValueError("temporal_reduction.output_space must be 'linear' or 'log2'")
        if isinstance(self.selection, EndpointSelection):
            if self.method != "identity":
                raise ValueError("endpoint temporal reductions require method 'identity'")
            if self.support.boundary_support != "none" or self.support.maximum_interior_gap_h is not None:
                raise ValueError("endpoint temporal reductions cannot declare interval boundary or gap support")
            if self.support.minimum_observations != 1:
                raise ValueError("endpoint temporal reductions require exactly one minimum observation")
        elif self.method == "identity":
            raise ValueError("interval temporal reductions cannot use method 'identity'")
        if self.method in {"observed_mean", "observed_median"} and self.support.minimum_observations < 1:
            raise ValueError(f"{self.method} requires at least one minimum observation")
        if self.method in {"geometric_time_mean", "integrated_linear_mean"}:
            if self.output_space != "log2":
                raise ValueError(f"{self.method} requires output_space 'log2'")
            if self.support.boundary_support == "none":
                raise ValueError(f"{self.method} requires covered or observed interval boundaries")
            if self.support.maximum_interior_gap_h is None or self.support.positive_floor is None:
                raise ValueError(f"{self.method} requires maximum_interior_gap_h and positive_floor")
        if self.output_space == "log2" and self.support.positive_floor is None:
            raise ValueError("log2 temporal reductions require support.positive_floor")

    @classmethod
    def from_mapping(cls, value: object) -> TemporalReductionSpec:
        payload = _exact_fields(
            value,
            context="temporal_reduction",
            required={"selection", "method", "output_space", "support"},
        )
        return cls(
            selection=parse_temporal_selection(payload["selection"]),
            method=_choice(
                payload["method"],
                context="temporal_reduction.method",
                allowed={
                    "identity",
                    "observed_mean",
                    "observed_median",
                    "geometric_time_mean",
                    "integrated_linear_mean",
                },
            ),  # type: ignore[arg-type]
            output_space=_choice(
                payload["output_space"],
                context="temporal_reduction.output_space",
                allowed={"linear", "log2"},
            ),  # type: ignore[arg-type]
            support=TemporalSupportPolicy.from_mapping(payload["support"]),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "selection": self.selection.to_mapping(),
            "method": self.method,
            "output_space": self.output_space,
            "support": self.support.to_mapping(),
        }


__all__ = [
    "EndpointSelection",
    "IntervalSelection",
    "TemporalReductionSpec",
    "TemporalSelection",
    "TemporalSupportPolicy",
    "parse_temporal_selection",
]
