from __future__ import annotations

from pathlib import Path
from typing import Any

from reader.core.errors import ExecutionError
from reader.workbench.graph import FileRef, InputRef, RecordRef, ResourceRef
from reader.workbench.ports import InputPortSpec
from reader.workbench.records import RecordStore


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
            path = target.path
            if not path.exists():
                hint = _missing_metadata_hint(label, path, exp_dir)
                if hint:
                    raise ExecutionError(f"Input file missing for '{label}': {path}. {hint}")
                raise ExecutionError(f"Input file missing for '{label}': {path}")
            if path.is_dir():
                raise ExecutionError(f"Input file path is a directory for '{label}': {path}")
            inputs[label] = path
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
