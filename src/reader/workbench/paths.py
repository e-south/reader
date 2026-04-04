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
