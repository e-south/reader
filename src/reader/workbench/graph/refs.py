from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordRef:
    record_id: str


@dataclass(frozen=True)
class FileRef:
    path: Path


@dataclass(frozen=True)
class ResourceRef:
    resource_id: str
    path: Path


InputRef = RecordRef | FileRef | ResourceRef


@dataclass(frozen=True)
class OutputRef:
    record_id: str


@dataclass(frozen=True)
class ProvenanceInput:
    label: str
    ref: InputRef
    discovery_policy: str | None = None


def input_ref_display(ref: InputRef) -> str:
    if isinstance(ref, RecordRef):
        return ref.record_id
    if isinstance(ref, ResourceRef):
        return f"resource({ref.resource_id})"
    return f"file({ref.path})"


def output_ref_display(ref: OutputRef) -> str:
    return ref.record_id


def input_ref_to_dict(ref: InputRef) -> dict[str, str]:
    if isinstance(ref, RecordRef):
        return {"record": ref.record_id}
    if isinstance(ref, ResourceRef):
        return {"resource": ref.resource_id, "path": str(ref.path)}
    return {"file": str(ref.path)}


def output_ref_to_dict(ref: OutputRef) -> dict[str, str]:
    return {"record": ref.record_id}


def provenance_input_to_dict(binding: ProvenanceInput) -> dict[str, str]:
    return {"label": binding.label, **input_ref_to_dict(binding.ref)}


def input_ref_from_dict(payload: dict[str, Any]) -> InputRef:
    if not isinstance(payload, dict):
        raise TypeError("input binding payload must be a mapping")
    keys = set(payload)
    if keys == {"record"}:
        record = payload["record"]
        if isinstance(record, str) and record:
            return RecordRef(record_id=record)
        raise ValueError("record binding must include a non-empty record id")
    if keys == {"file"}:
        file_path = payload["file"]
        if isinstance(file_path, str) and file_path:
            return FileRef(path=Path(file_path))
        raise ValueError("file binding must include a non-empty path")
    if keys == {"resource", "path"}:
        resource_id = payload["resource"]
        path = payload["path"]
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("resource binding must include a non-empty resource id")
        if not isinstance(path, str) or not path:
            raise ValueError(f"resource binding {resource_id!r} must include a non-empty path")
        return ResourceRef(resource_id=resource_id, path=Path(path))
    raise ValueError(
        "input binding payload must contain exactly one reference shape: {'record'}, {'file'}, or {'resource', 'path'}"
    )


def output_ref_from_dict(payload: dict[str, Any]) -> OutputRef:
    if not isinstance(payload, dict):
        raise TypeError("output binding payload must be a mapping")
    if set(payload) != {"record"}:
        raise ValueError("output binding payload must contain only 'record'")
    record = payload.get("record")
    if not isinstance(record, str) or not record:
        raise ValueError("output binding payload must include record")
    return OutputRef(record_id=record)


def provenance_input_from_dict(payload: dict[str, Any]) -> ProvenanceInput:
    if not isinstance(payload, dict):
        raise TypeError("provenance input payload must be a mapping")
    label = payload.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("provenance input payload must include label")
    ref_payload = dict(payload)
    ref_payload.pop("label", None)
    return ProvenanceInput(label=label, ref=input_ref_from_dict(ref_payload))
