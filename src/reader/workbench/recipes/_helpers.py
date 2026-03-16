from __future__ import annotations

from typing import Any

from reader.core.errors import ConfigError
from reader.workbench.decl import (
    FileInputDecl,
    PluginStepDecl,
    RecordInputDecl,
    RecordOutputDecl,
    ResourceInputDecl,
)


def recipe_step(
    *,
    id: str,
    plugin: str,
    reads: dict[str, dict[str, str]] | None = None,
    with_: dict[str, Any] | None = None,
    writes: dict[str, dict[str, str]] | None = None,
) -> PluginStepDecl:
    return PluginStepDecl(
        id=id,
        plugin=plugin,
        reads={key: _parse_input_decl(value, label=key) for key, value in (reads or {}).items()},
        with_=dict(with_ or {}),
        writes={key: _parse_output_decl(value, label=key) for key, value in (writes or {}).items()},
    )


def _parse_input_decl(raw: dict[str, str], *, label: str):
    if not isinstance(raw, dict):
        raise ConfigError(f"Recipe reads.{label} must be a mapping")
    populated = [key for key in ("record", "file", "resource") if raw.get(key) is not None]
    if len(populated) != 1:
        raise ConfigError(f"Recipe reads.{label} must declare exactly one of record, file, or resource")
    key = populated[0]
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Recipe reads.{label}.{key} must be a non-empty string")
    if key == "record":
        return RecordInputDecl(record_id=value)
    if key == "file":
        return FileInputDecl(path=value)
    return ResourceInputDecl(resource_id=value)


def _parse_output_decl(raw: dict[str, str], *, label: str) -> RecordOutputDecl:
    if not isinstance(raw, dict):
        raise ConfigError(f"Recipe writes.{label} must be a mapping")
    value = raw.get("record")
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Recipe writes.{label}.record must be a non-empty string")
    return RecordOutputDecl(record_id=value)
