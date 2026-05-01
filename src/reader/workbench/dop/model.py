from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


def _clean_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _clean_tuple(values: Iterable[str], *, field_name: str, allow_empty: bool = False) -> tuple[str, ...]:
    cleaned = tuple(str(value).strip() for value in values if str(value).strip())
    if not allow_empty and not cleaned:
        raise ValueError(f"{field_name} must include at least one value.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field_name} must not include duplicate values.")
    return cleaned


@dataclass(frozen=True)
class DataClassSpec:
    id: str
    label: str
    summary: str
    decision_order: int
    protocol_candidates: tuple[str, ...]
    minimum_capture: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    transfer_rules: tuple[str, ...]
    verification: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _clean_string(self.id, field_name="DataClassSpec.id"))
        object.__setattr__(self, "label", _clean_string(self.label, field_name="DataClassSpec.label"))
        object.__setattr__(self, "summary", _clean_string(self.summary, field_name="DataClassSpec.summary"))
        if not isinstance(self.decision_order, int) or self.decision_order < 0:
            raise ValueError("DataClassSpec.decision_order must be a non-negative integer.")
        object.__setattr__(
            self,
            "protocol_candidates",
            _clean_tuple(self.protocol_candidates, field_name="DataClassSpec.protocol_candidates"),
        )
        object.__setattr__(
            self,
            "minimum_capture",
            _clean_tuple(self.minimum_capture, field_name="DataClassSpec.minimum_capture"),
        )
        object.__setattr__(
            self,
            "stop_conditions",
            _clean_tuple(self.stop_conditions, field_name="DataClassSpec.stop_conditions"),
        )
        object.__setattr__(
            self,
            "transfer_rules",
            _clean_tuple(self.transfer_rules, field_name="DataClassSpec.transfer_rules"),
        )
        object.__setattr__(
            self,
            "verification",
            _clean_tuple(self.verification, field_name="DataClassSpec.verification"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "summary": self.summary,
            "decision_order": self.decision_order,
            "protocol_candidates": list(self.protocol_candidates),
            "minimum_capture": list(self.minimum_capture),
            "stop_conditions": list(self.stop_conditions),
            "transfer_rules": list(self.transfer_rules),
            "verification": list(self.verification),
        }


@dataclass(frozen=True)
class ReadySpec:
    id: str
    label: str
    summary: str
    required_evidence: tuple[str, ...]
    accepted_readiness_states: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _clean_string(self.id, field_name="ReadySpec.id"))
        object.__setattr__(self, "label", _clean_string(self.label, field_name="ReadySpec.label"))
        object.__setattr__(self, "summary", _clean_string(self.summary, field_name="ReadySpec.summary"))
        object.__setattr__(
            self,
            "required_evidence",
            _clean_tuple(self.required_evidence, field_name="ReadySpec.required_evidence"),
        )
        object.__setattr__(
            self,
            "accepted_readiness_states",
            _clean_tuple(
                self.accepted_readiness_states,
                field_name="ReadySpec.accepted_readiness_states",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "required_capabilities",
            _clean_tuple(
                self.required_capabilities,
                field_name="ReadySpec.required_capabilities",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "commands",
            _clean_tuple(self.commands, field_name="ReadySpec.commands", allow_empty=True),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "summary": self.summary,
            "required_evidence": list(self.required_evidence),
            "accepted_readiness_states": list(self.accepted_readiness_states),
            "required_capabilities": list(self.required_capabilities),
            "commands": list(self.commands),
        }


class DopRegistry:
    def __init__(self, *, data_classes: Iterable[DataClassSpec], ready_specs: Iterable[ReadySpec]):
        data_class_items = tuple(sorted(data_classes, key=lambda item: item.decision_order))
        ready_spec_items = tuple(ready_specs)
        self._data_classes = data_class_items
        self._ready_specs = ready_spec_items
        self._data_classes_by_id = _index_by_id(data_class_items, kind="DOP data class")
        self._ready_specs_by_id = _index_by_id(ready_spec_items, kind="DOP ready spec")
        orders = [item.decision_order for item in data_class_items]
        if len(set(orders)) != len(orders):
            raise ValueError("DOP data class decision_order values must be unique.")

    def data_classes(self) -> tuple[DataClassSpec, ...]:
        return self._data_classes

    def ready_specs(self) -> tuple[ReadySpec, ...]:
        return self._ready_specs

    def data_class(self, data_class_id: str) -> DataClassSpec:
        key = str(data_class_id).strip()
        try:
            return self._data_classes_by_id[key]
        except KeyError:
            options = ", ".join(sorted(self._data_classes_by_id)) or "—"
            raise ValueError(f"Unknown DOP data class {data_class_id!r}. Available classes: {options}") from None

    def ready_spec(self, ready_spec_id: str) -> ReadySpec:
        key = str(ready_spec_id).strip()
        try:
            return self._ready_specs_by_id[key]
        except KeyError:
            options = ", ".join(sorted(self._ready_specs_by_id)) or "—"
            raise ValueError(f"Unknown DOP ready spec {ready_spec_id!r}. Available specs: {options}") from None

    def data_classes_for_protocol(self, protocol_id: str) -> tuple[DataClassSpec, ...]:
        key = str(protocol_id).strip()
        return tuple(item for item in self._data_classes if key in item.protocol_candidates)

    def validate_protocol_refs(self, protocol_ids: Iterable[str]) -> None:
        known = {str(protocol_id).strip() for protocol_id in protocol_ids if str(protocol_id).strip()}
        referenced = {
            protocol_id for data_class in self._data_classes for protocol_id in data_class.protocol_candidates
        }
        missing = sorted(referenced - known)
        if missing:
            raise ValueError("DOP registry references unknown protocol ids: " + ", ".join(missing))

    def validate_ready_refs(self, *, readiness_states: Iterable[str], capability_keys: Iterable[str]) -> None:
        known_states = {str(state).strip() for state in readiness_states if str(state).strip()}
        known_capabilities = {str(capability).strip() for capability in capability_keys if str(capability).strip()}
        missing_states = sorted(
            state for spec in self._ready_specs for state in spec.accepted_readiness_states if state not in known_states
        )
        missing_capabilities = sorted(
            capability
            for spec in self._ready_specs
            for capability in spec.required_capabilities
            if capability not in known_capabilities
        )
        errors = []
        if missing_states:
            errors.append("unknown readiness states: " + ", ".join(missing_states))
        if missing_capabilities:
            errors.append("unknown readiness capabilities: " + ", ".join(missing_capabilities))
        if errors:
            raise ValueError("DOP ready specs reference " + "; ".join(errors))


def _index_by_id(items: Iterable[DataClassSpec] | Iterable[ReadySpec], *, kind: str):
    indexed = {}
    for item in items:
        if item.id in indexed:
            raise ValueError(f"Duplicate {kind} id {item.id!r}.")
        indexed[item.id] = item
    return indexed
