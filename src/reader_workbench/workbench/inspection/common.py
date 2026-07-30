from __future__ import annotations

import json
from pathlib import Path


def count_visible_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and not item.name.startswith("."))


def count_visible_glob(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob(pattern) if item.is_file() and not item.name.startswith("."))


def visible_relative_files(path: Path, *, base: Path, limit: int = 8) -> list[str]:
    if not path.exists():
        return []
    files = sorted(item for item in path.rglob("*") if item.is_file() and not item.name.startswith("."))
    return [format_relative_path(item, base=base) for item in files[:limit]]


def format_relative_path(path: Path, *, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def resolve_output_subdir(outputs_dir: Path, subdir: str) -> Path:
    return outputs_dir if subdir in ("", ".", "./") else outputs_dir / subdir


def summarize_outputs_dir(
    outputs_dir: Path,
    *,
    plots_subdir: str = "plots",
    exports_subdir: str = "exports",
    notebooks_subdir: str = "notebooks",
) -> dict[str, int]:
    plots_dir = resolve_output_subdir(outputs_dir, plots_subdir)
    exports_dir = resolve_output_subdir(outputs_dir, exports_subdir)
    notebooks_dir = resolve_output_subdir(outputs_dir, notebooks_subdir)
    artifacts_dir = outputs_dir / "artifacts"
    return {
        "records": count_visible_glob(artifacts_dir, "*.parquet"),
        "plots": count_visible_files(plots_dir),
        "exports": count_visible_files(exports_dir),
        "notebooks": count_visible_files(notebooks_dir),
    }


def preview_output_files(path: Path, *, base: Path, limit: int = 4) -> str:
    files = visible_relative_files(path, base=base, limit=limit)
    if not files:
        return "—"
    remaining = count_visible_files(path) - len(files)
    preview = ", ".join(files)
    if remaining > 0:
        preview += f", … (+{remaining} more)"
    return preview


def render_compact_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        value = list(value)
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return str(value)


def flatten_binding_rows(value, *, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child_path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_binding_rows(value[key], prefix=child_path))
        return rows
    rows.append((prefix or "value", render_compact_value(value)))
    return rows
