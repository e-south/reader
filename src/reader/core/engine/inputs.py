from __future__ import annotations

from pathlib import Path
from typing import Any

from reader.core.errors import ExecutionError
from reader.core.records import RecordStore


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
    hint = f"Canonical location is {canonical} (update config: reads.{label}: file:./inputs/{path.name})."

    candidates = _metadata_like_files(inputs_dir)
    if candidates:
        rels = [str(candidate.relative_to(exp_dir)) for candidate in candidates]
        preview = ", ".join(rels[:3])
        tail = "" if len(rels) <= 3 else f" (+{len(rels) - 3} more)"
        hint += f" Found metadata-like files in inputs/: {preview}{tail}."
    return hint


def _resolve_inputs(store: RecordStore, reads: dict[str, str], *, exp_dir: Path | None = None) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for label, target in reads.items():
        if isinstance(target, str) and target.startswith("file:"):
            path = Path(target.split("file:", 1)[1])
            if not path.exists():
                hint = _missing_metadata_hint(label, path, exp_dir)
                if hint:
                    raise ExecutionError(f"Input file missing for '{label}': {path}. {hint}")
                raise ExecutionError(f"Input file missing for '{label}': {path}")
            if path.is_dir():
                raise ExecutionError(f"Input file path is a directory for '{label}': {path}")
            inputs[label] = path
            continue
        inputs[label] = store.read_dataframe(target)
    return inputs
