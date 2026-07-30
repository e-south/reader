from __future__ import annotations

from pathlib import Path

import typer
import yaml

from reader_workbench.errors import ConfigError
from reader_workbench.workbench.experiment import ResourceCatalog
from reader_workbench.workbench.graph import (
    FileRef,
    InputRef,
    OutputRef,
    RecordRef,
    ResourceRef,
    select_workbench_specs,
)
from reader_workbench.workbench.paths import resolve_path_within_root


def build_surface_command(
    command: str,
    job_path: Path,
    *,
    only: list[str] | None,
    exclude: list[str] | None,
    list_only: bool = False,
    dry_run: bool = False,
    log_level: str = "INFO",
    inputs: list[str] | None = None,
    sets: list[str] | None = None,
) -> list[str]:
    parts = ["uv", "run", *command.split(), str(job_path)]
    if list_only:
        parts += ["--list"]
    if only:
        for value in only:
            parts += ["--only", value]
    if exclude:
        for value in exclude:
            parts += ["--exclude", value]
    if dry_run:
        parts += ["--dry-run"]
    if log_level and log_level != "INFO":
        parts += ["--log-level", log_level]
    for raw in inputs or []:
        parts += ["--input", raw]
    for raw in sets or []:
        parts += ["--set", raw]
    return parts


def select_surface_specs(steps, *, only: list[str], exclude: list[str], kind: str):
    try:
        return select_workbench_specs(steps, only=only, exclude=exclude, kind_label=kind)
    except ConfigError as err:
        raise typer.BadParameter(f"{err} Use --list to see valid ids.") from err


def parse_input_overrides(
    raw_inputs: list[str],
    *,
    root: Path,
    resources: ResourceCatalog,
) -> dict[str, InputRef]:
    overrides: dict[str, InputRef] = {}
    for raw in raw_inputs:
        if "=" not in raw:
            raise typer.BadParameter("--input expects KEY=VALUE")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise typer.BadParameter("--input key cannot be empty")
        if not value:
            raise typer.BadParameter("--input value cannot be empty")
        overrides[key] = _coerce_cli_input_ref(yaml.safe_load(value), root=root, resources=resources)
    return overrides


def parse_set_overrides(raw_sets: list[str]) -> list[tuple[str, object]]:
    overrides: list[tuple[str, object]] = []
    for raw in raw_sets:
        if "=" not in raw:
            raise typer.BadParameter("--set expects PATH=VALUE")
        path, value_raw = raw.split("=", 1)
        path = path.strip()
        if not path:
            raise typer.BadParameter("--set path cannot be empty")
        overrides.append((path, yaml.safe_load(value_raw)))
    return overrides


def apply_step_overrides(
    steps,
    *,
    input_overrides: dict[str, InputRef],
    set_overrides: list[tuple[str, object]],
    root: Path,
    resources: ResourceCatalog,
):
    updated = []
    for step in steps:
        if hasattr(step, "model_copy"):
            cloned = step.model_copy(deep=True)
            reads = dict(cloned.reads or {})
            with_block = dict(cloned.with_ or {})
            writes = dict(cloned.writes or {})
        else:
            reads = dict(step.reads or {})
            with_block = dict(step.with_ or {})
            writes = dict(step.writes or {})
        if input_overrides:
            reads.update(input_overrides)
        for path, value in set_overrides:
            parts = [item for item in path.split(".") if item]
            if not parts:
                raise typer.BadParameter("--set path cannot be empty")
            section = parts[0]
            if section not in {"reads", "with", "writes"}:
                raise typer.BadParameter("--set path must start with reads., with., or writes.")
            if section in {"reads", "writes"}:
                if len(parts) != 2:
                    raise typer.BadParameter(f"--set {section} expects a single key (e.g., {section}.foo=bar)")
                target = reads if section == "reads" else writes
                target[parts[1]] = (
                    _coerce_cli_input_ref(value, root=root, resources=resources)
                    if section == "reads"
                    else _coerce_cli_output_ref(value)
                )
            else:
                if len(parts) < 2:
                    raise typer.BadParameter("--set with.* requires a key (e.g., with.foo=bar)")
                _set_nested(with_block, parts[1:], value)
        if hasattr(step, "model_copy"):
            cloned.reads = reads
            cloned.with_ = with_block
            cloned.writes = writes
            updated.append(cloned)
            continue
        payload = {
            "id": step.id,
            "plugin": step.plugin,
            "reads": reads,
            "with_": with_block,
            "writes": writes,
            "source_recipe": getattr(step, "source_recipe", None),
        }
        if hasattr(step, "kind"):
            payload["kind"] = step.kind
        updated.append(step.__class__(**payload))
    return updated


def _coerce_cli_input_ref(value, *, root: Path, resources: ResourceCatalog) -> InputRef:
    if isinstance(value, (RecordRef, FileRef, ResourceRef)):
        return value
    if isinstance(value, dict):
        record = value.get("record")
        file_path = value.get("file")
        resource_id = value.get("resource")
        populated = [item for item in (record, file_path, resource_id) if item is not None]
        if len(populated) != 1:
            raise typer.BadParameter("reads.* must declare exactly one of record, file, or resource")
        if isinstance(record, str) and record.strip():
            return RecordRef(record_id=record.strip())
        if isinstance(file_path, str) and file_path.strip():
            path = Path(file_path.strip()).expanduser()
            try:
                path = resolve_path_within_root(path, root=root)
            except ValueError as err:
                raise typer.BadParameter(
                    "reads.* file bindings must stay under the experiment root after resolving symlinks"
                ) from err
            return FileRef(path=path)
        if isinstance(resource_id, str) and resource_id.strip():
            return _resolve_cli_resource_ref(resource_id.strip(), resources=resources)
        raise typer.BadParameter("reads.* binding values must be non-empty strings")
    if isinstance(value, str) and value.strip():
        return RecordRef(record_id=value.strip())
    raise typer.BadParameter("reads.* expects a YAML/JSON mapping like {record: ...}, {file: ...}, or {resource: ...}")


def _resolve_cli_resource_ref(resource_id: str, *, resources: ResourceCatalog) -> ResourceRef:
    if not resource_id:
        raise typer.BadParameter("resource bindings require a non-empty resource id")
    try:
        resource = resources.require_file(resource_id)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err
    return ResourceRef(resource_id=resource_id, path=resource.path.resolve())


def _coerce_cli_output_ref(value) -> OutputRef:
    if isinstance(value, OutputRef):
        return value
    if isinstance(value, dict):
        record = value.get("record")
        if isinstance(record, str) and record.strip():
            return OutputRef(record_id=record.strip())
        raise typer.BadParameter("writes.* must declare {record: ...}")
    if isinstance(value, str) and value.strip():
        return OutputRef(record_id=value.strip())
    raise typer.BadParameter("writes.* expects a record id or a {record: ...} mapping")


def _set_nested(mapping: dict, keys: list[str], value) -> None:
    current = mapping
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        if not isinstance(current[key], dict):
            raise typer.BadParameter(f"--set path invalid (non-mapping at '{key}')")
        current = current[key]
    current[keys[-1]] = value
