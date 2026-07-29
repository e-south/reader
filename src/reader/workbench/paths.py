from __future__ import annotations

from pathlib import Path


def resolve_path_within_root(raw: str | Path, *, root: Path) -> Path:
    root_path = root.resolve()
    raw_path = Path(raw).expanduser()
    resolved = (root_path / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as err:
        raise ValueError(f"{resolved} escapes {root_path}") from err
    return resolved


def resolve_confined_sink_root(path: str | Path, *, root: Path, label: str) -> Path:
    """Validate a writable sink root without following symlinked path components."""

    root_path = Path(root).expanduser().absolute()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root_path / candidate
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"{label} sink root must stay within {root_path}") from exc

    cursor = root_path
    if cursor.is_symlink():
        raise ValueError(f"{label} sink root must not use symlink path components: {cursor}")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} sink root must not use symlink path components: {cursor}")

    resolved_root = root_path.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} sink root must resolve within {resolved_root}") from exc
    return candidate
