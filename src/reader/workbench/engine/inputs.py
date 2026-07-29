from __future__ import annotations

from pathlib import Path
from typing import Any

from reader.errors import ExecutionError
from reader.workbench.graph import FileRef, InputRef, RecordCollectionRef, RecordRef, ResourceRef
from reader.workbench.paths import resolve_path_within_root
from reader.workbench.ports import InputPortSpec
from reader.workbench.records import RecordStore, SourceRecordCollection, resolve_source_record
from reader.workbench.registry import Plugin, PluginConfig


def _metadata_like_files(inputs_dir: Path) -> list[Path]:
    patterns = [
        "metadata.*",
        "metadata_filtered.*",
        "sample_map.*",
        "sample_metadata.*",
        "plate_map.*",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in inputs_dir.glob(pattern) if path.is_file())
    return sorted(matches)


def _missing_metadata_hint(label: str, path: Path, exp_dir: Path | None) -> str | None:
    if exp_dir is None:
        return None
    inputs_dir = exp_dir / "inputs"
    if not inputs_dir.exists():
        return None

    metadata_labels = {"sample_map", "plate_map", "metadata", "sample_metadata"}
    name = path.name.lower()
    is_metadataish = label in metadata_labels or name.startswith(("metadata", "sample_map", "plate_map"))
    if not is_metadataish:
        return None

    canonical = inputs_dir / path.name
    hint = f"Canonical location is {canonical} (update config: reads.{label}: {{file: ./inputs/{path.name}}})."

    candidates = _metadata_like_files(inputs_dir)
    if candidates:
        rels = [str(candidate.relative_to(exp_dir)) for candidate in candidates]
        preview = ", ".join(rels[:3])
        tail = "" if len(rels) <= 3 else f" (+{len(rels) - 3} more)"
        hint += f" Found metadata-like files in inputs/: {preview}{tail}."
    return hint


def _resolve_inputs(
    store: RecordStore,
    reads: dict[str, InputRef],
    *,
    input_ports: dict[str, InputPortSpec],
    exp_dir: Path | None = None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for label, target in reads.items():
        expected = input_ports.get(label)
        if expected is None:
            raise ExecutionError(f"Unsupported input binding for unknown port '{label}'")
        if isinstance(target, FileRef | ResourceRef):
            if expected.kind not in {"file_path", "file_set"}:
                raise ExecutionError(
                    f"Input '{label}' expects port kind {expected.kind!r} and cannot bind to a file/resource ref"
                )
            path = target.path
            if exp_dir is not None:
                try:
                    path = resolve_path_within_root(path, root=exp_dir)
                except ValueError as exc:
                    raise ExecutionError(
                        f"Input file for '{label}' must stay under the experiment root after resolving symlinks"
                    ) from exc
            if not path.exists():
                hint = _missing_metadata_hint(label, path, exp_dir)
                if hint:
                    raise ExecutionError(f"Input file missing for '{label}': {path}. {hint}")
                raise ExecutionError(f"Input file missing for '{label}': {path}")
            if path.is_dir():
                raise ExecutionError(f"Input file path is a directory for '{label}': {path}")
            inputs[label] = (path,) if expected.kind == "file_set" else path
            continue
        if isinstance(target, RecordCollectionRef):
            if expected.kind != "record_collection":
                raise ExecutionError(
                    f"Input '{label}' expects port kind {expected.kind!r} and cannot bind to a record collection"
                )
            inputs[label] = SourceRecordCollection(
                tuple(resolve_source_record(ref, contracts=store.contracts) for ref in target.records)
            )
            continue
        if not isinstance(target, RecordRef):
            raise ExecutionError(f"Unsupported input binding for '{label}': {target!r}")
        if expected.kind == "dataframe":
            inputs[label] = store.read_dataframe(target.record_id)
            continue
        if expected.kind == "file_bundle":
            inputs[label] = store.read_record(target.record_id)
            continue
        raise ExecutionError(
            f"Input '{label}' expects port kind {expected.kind!r} and cannot bind to a record ref {target.record_id!r}"
        )
    return inputs


def resolve_missing_file_inputs(
    *,
    plugin: Plugin,
    exp_dir: Path,
    cfg: PluginConfig,
    inputs: dict[str, Any],
    input_ports: dict[str, InputPortSpec],
) -> dict[str, Any]:
    try:
        additions = dict(type(plugin).resolve_missing_file_inputs(exp_dir=exp_dir, cfg=cfg, inputs=inputs))
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError(f"{plugin.plugin_id}: failed to resolve missing file inputs: {exc}") from exc
    conflicts = sorted(set(additions) & set(inputs))
    if conflicts:
        raise ExecutionError(f"{plugin.plugin_id}: missing-file resolver cannot replace bound inputs: {conflicts}")
    unknown = sorted(set(additions) - set(input_ports))
    if unknown:
        raise ExecutionError(f"{plugin.plugin_id}: missing-file resolver returned unknown inputs: {unknown}")

    resolved = dict(inputs)
    for label, value in additions.items():
        port = input_ports[label]
        if port.kind not in {"file_path", "file_set"} or not port.optional:
            raise ExecutionError(
                f"{plugin.plugin_id}: missing-file resolver may only fill optional file_path or file_set ports; "
                f"{label!r} is {port.kind!r} (optional={port.optional})"
            )
        values = value if port.kind == "file_set" else (value,)
        if not isinstance(values, tuple) or not values or any(not isinstance(item, Path) for item in values):
            expected = "a non-empty tuple of Paths" if port.kind == "file_set" else "a Path"
            raise ExecutionError(
                f"{plugin.plugin_id}: missing-file resolver input {label!r} must be {expected}, "
                f"got {type(value).__name__}"
            )
        confined_values: list[Path] = []
        for item in values:
            try:
                confined = resolve_path_within_root(item, root=exp_dir)
            except ValueError as exc:
                raise ExecutionError(
                    f"{plugin.plugin_id}: resolved input {label!r} must stay under the experiment root "
                    "after resolving symlinks"
                ) from exc
            if not confined.exists():
                raise ExecutionError(f"Resolved input file missing for {label!r}: {confined}")
            if not confined.is_file():
                raise ExecutionError(f"Resolved input path is not a file for {label!r}: {confined}")
            confined_values.append(confined)
        resolved[label] = tuple(confined_values) if port.kind == "file_set" else confined_values[0]
    return resolved
