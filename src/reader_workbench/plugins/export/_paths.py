from __future__ import annotations

from pathlib import Path

from reader_workbench.errors import ExecutionError
from reader_workbench.workbench.paths import resolve_path_within_root


def resolve_export_path(raw: str, *, exports_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        raise ExecutionError("Export paths must be relative to the experiment exports directory")
    try:
        resolved = resolve_path_within_root(path, root=exports_dir)
    except ValueError as exc:
        raise ExecutionError(
            "Export paths must stay under the experiment exports directory after resolving symlinks"
        ) from exc
    if resolved == exports_dir.resolve():
        raise ExecutionError("Export paths must identify a file below the experiment exports directory")
    return resolved
