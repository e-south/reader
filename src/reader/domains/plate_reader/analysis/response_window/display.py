"""Strict display vocabulary for generic response-window review surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

DISPLAY_SCHEMA_VERSION = "reader.response_window.display.v1"
STATE_ORDER = ("00", "10", "01", "11")
ExampleRole = Literal["reference_anchor", "response_example"]


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _exact(value: object, *, context: str, fields: set[str]) -> dict[str, Any]:
    payload = _mapping(value, context=context)
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        raise ValueError(f"{context} fields disagree: missing={missing}, unknown={unknown}.")
    return payload


def _text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True)
class DisplayExample:
    design_id: str
    label: str
    role: ExampleRole

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> DisplayExample:
        payload = _exact(value, context=f"display.examples[{index}]", fields={"design_id", "label", "role"})
        role = _text(payload["role"], context=f"display.examples[{index}].role")
        if role not in {"reference_anchor", "response_example"}:
            raise ValueError(f"display.examples[{index}].role is unsupported: {role!r}.")
        return cls(
            design_id=_text(payload["design_id"], context=f"display.examples[{index}].design_id"),
            label=_text(payload["label"], context=f"display.examples[{index}].label"),
            role=role,
        )

    def to_mapping(self) -> dict[str, str]:
        return {"design_id": self.design_id, "label": self.label, "role": self.role}


@dataclass(frozen=True)
class ResponseWindowDisplaySpec:
    study_label: str
    event_label: str
    state_labels: dict[str, str]
    examples: tuple[DisplayExample, ...]

    @property
    def reference_anchor(self) -> DisplayExample:
        return next(example for example in self.examples if example.role == "reference_anchor")

    @classmethod
    def from_mapping(cls, value: object) -> ResponseWindowDisplaySpec:
        payload = _exact(
            value,
            context="display",
            fields={"study_label", "event_label", "state_labels", "examples"},
        )
        raw_labels = _mapping(payload["state_labels"], context="display.state_labels")
        if set(raw_labels) != set(STATE_ORDER):
            raise ValueError(f"display.state_labels must define exactly {list(STATE_ORDER)!r}.")
        state_labels = {
            state: _text(raw_labels[state], context=f"display.state_labels.{state}") for state in STATE_ORDER
        }
        if len(set(state_labels.values())) != len(STATE_ORDER):
            raise ValueError("display.state_labels must be unique.")
        raw_examples = payload["examples"]
        if isinstance(raw_examples, (str, bytes)) or not isinstance(raw_examples, Sequence):
            raise ValueError("display.examples must be a non-empty sequence.")
        examples = tuple(DisplayExample.from_mapping(item, index=index) for index, item in enumerate(raw_examples))
        if not examples or len({item.design_id for item in examples}) != len(examples):
            raise ValueError("display.examples must be non-empty with unique design ids.")
        if sum(item.role == "reference_anchor" for item in examples) != 1:
            raise ValueError("display.examples must define exactly one reference anchor.")
        if not any(item.role == "response_example" for item in examples):
            raise ValueError("display.examples must define at least one response example.")
        return cls(
            study_label=_text(payload["study_label"], context="display.study_label"),
            event_label=_text(payload["event_label"], context="display.event_label"),
            state_labels=state_labels,
            examples=examples,
        )

    def to_manifest(
        self,
        *,
        response_ratio: str,
        magnitude_ratio: str,
        growth: str,
        reference_design_id: str,
    ) -> dict[str, object]:
        return {
            "schema_version": DISPLAY_SCHEMA_VERSION,
            "study_label": self.study_label,
            "event_label": self.event_label,
            "state_labels": dict(self.state_labels),
            "channels": {
                "response_ratio": response_ratio,
                "magnitude_ratio": magnitude_ratio,
                "growth": growth,
                "reference_design_id": reference_design_id,
            },
            "examples": [example.to_mapping() for example in self.examples],
        }


def validate_display_manifest(value: object) -> dict[str, object]:
    payload = _exact(
        value,
        context="bundle.display",
        fields={"schema_version", "study_label", "event_label", "state_labels", "channels", "examples"},
    )
    if payload["schema_version"] != DISPLAY_SCHEMA_VERSION:
        raise ValueError(f"bundle.display must use {DISPLAY_SCHEMA_VERSION!r}.")
    spec = ResponseWindowDisplaySpec.from_mapping(
        {
            "study_label": payload["study_label"],
            "event_label": payload["event_label"],
            "state_labels": payload["state_labels"],
            "examples": payload["examples"],
        }
    )
    channels = _exact(
        payload["channels"],
        context="bundle.display.channels",
        fields={"response_ratio", "magnitude_ratio", "growth", "reference_design_id"},
    )
    normalized_channels = {
        key: _text(value, context=f"bundle.display.channels.{key}") for key, value in channels.items()
    }
    if spec.reference_anchor.design_id != normalized_channels["reference_design_id"]:
        raise ValueError("bundle display reference anchor disagrees with its reference design id.")
    return {
        "schema_version": DISPLAY_SCHEMA_VERSION,
        "study_label": spec.study_label,
        "event_label": spec.event_label,
        "state_labels": dict(spec.state_labels),
        "channels": normalized_channels,
        "examples": [example.to_mapping() for example in spec.examples],
    }


__all__ = [
    "DISPLAY_SCHEMA_VERSION",
    "DisplayExample",
    "ResponseWindowDisplaySpec",
    "validate_display_manifest",
]
