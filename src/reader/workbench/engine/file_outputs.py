from __future__ import annotations

import re
import shutil
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from reader.errors import ExecutionError
from reader.workbench.context import RunContext
from reader.workbench.records import PathDescription

from .contracts import collect_file_output_paths

_SAFE_PREFIX = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class _Promotion:
    final_path: Path
    backup_path: Path
    previous_exists: bool


class FileOutputTransaction:
    """Stage one plot/export step and restore prior files unless it commits."""

    def __init__(self, *, context: RunContext, step_id: str) -> None:
        self._final_context = context
        staging_parent = context.outputs_dir / ".staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        prefix = _SAFE_PREFIX.sub("_", step_id).strip("._") or "file-output"
        self._staging_root = Path(mkdtemp(prefix=f"{prefix}__", dir=staging_parent))
        self.context = replace(
            context,
            outputs_dir=self._staging_root,
            artifacts_dir=self._mirror_path(context.artifacts_dir),
            plots_dir=self._mirror_path(context.plots_dir),
            exports_dir=self._mirror_path(context.exports_dir),
            records_path=self._mirror_path(context.records_path),
        )
        self.context.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.context.plots_dir.mkdir(parents=True, exist_ok=True)
        self.context.exports_dir.mkdir(parents=True, exist_ok=True)
        self.context.records_path.parent.mkdir(parents=True, exist_ok=True)
        self._promotions: list[_Promotion] = []
        self._committed = False

    def __enter__(self) -> FileOutputTransaction:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if not self._committed:
                self._rollback()
        finally:
            shutil.rmtree(self._staging_root, ignore_errors=True)
            with suppress(OSError):
                self._staging_root.parent.rmdir()
        return False

    def promote(
        self,
        *,
        outputs: dict[str, Any],
        output_ports: dict[str, Any],
        where: str,
    ) -> dict[str, Any]:
        staged_paths = collect_file_output_paths(output_ports=output_ports, outputs=outputs, where=where)
        if not staged_paths:
            raise ExecutionError(f"{where}: must emit at least one explicit file output")
        path_map = self._path_map(staged_paths, where=where)
        rollback_root = self._staging_root / ".rollback"

        for staged_path, final_path in path_map.items():
            relative_path = final_path.relative_to(self._final_context.outputs_dir)
            backup_path = rollback_root / relative_path
            previous_exists = final_path.exists()
            promotion = _Promotion(
                final_path=final_path,
                backup_path=backup_path,
                previous_exists=previous_exists,
            )
            self._promotions.append(promotion)

            final_path.parent.mkdir(parents=True, exist_ok=True)
            if previous_exists:
                if not final_path.is_file() or final_path.is_symlink():
                    raise ExecutionError(f"{where}: output target must be a regular file: {final_path}")
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                final_path.replace(backup_path)
            staged_path.replace(final_path)

        return {
            name: _remap_output_value(value, path_map=path_map)
            if output_ports[name].kind in {"file_path", "file_bundle"}
            else value
            for name, value in outputs.items()
        }

    def commit(self) -> None:
        self._committed = True

    def _mirror_path(self, path: Path) -> Path:
        try:
            relative_path = path.relative_to(self._final_context.outputs_dir)
        except ValueError as exc:
            raise ExecutionError(
                f"Runtime output path must stay within {self._final_context.outputs_dir}: {path}"
            ) from exc
        return self._staging_root / relative_path

    def _path_map(self, paths: list[Path], *, where: str) -> dict[Path, Path]:
        staging_root = self._staging_root.resolve(strict=True)
        final_root = self._final_context.outputs_dir.resolve(strict=True)
        path_map: dict[Path, Path] = {}
        final_paths: set[Path] = set()
        for raw_path in paths:
            staged_path = raw_path if raw_path.is_absolute() else staging_root / raw_path
            if staged_path.is_symlink():
                raise ExecutionError(f"{where}: staged output must be a regular file: {staged_path}")
            try:
                resolved = staged_path.resolve(strict=True)
                relative_path = resolved.relative_to(staging_root)
            except (OSError, ValueError) as exc:
                raise ExecutionError(f"{where}: file outputs must stay within the staged outputs directory") from exc
            if not resolved.is_file():
                raise ExecutionError(f"{where}: staged output must be a regular file: {staged_path}")
            final_path = final_root / relative_path
            if final_path in final_paths:
                raise ExecutionError(f"{where}: duplicate file output target: {final_path}")
            path_map[Path(raw_path)] = final_path
            final_paths.add(final_path)
        return path_map

    def _rollback(self) -> None:
        for promotion in reversed(self._promotions):
            if promotion.previous_exists:
                if not promotion.backup_path.exists():
                    continue
                promotion.final_path.unlink(missing_ok=True)
                promotion.final_path.parent.mkdir(parents=True, exist_ok=True)
                promotion.backup_path.replace(promotion.final_path)
            else:
                promotion.final_path.unlink(missing_ok=True)
        self._promotions.clear()


def _remap_output_value(value: Any, *, path_map: dict[Path, Path]) -> Any:
    if isinstance(value, PathDescription):
        return PathDescription(path=path_map[value.path], description=value.description)
    if isinstance(value, str):
        return str(path_map[Path(value)])
    if isinstance(value, Path):
        return path_map[value]
    if isinstance(value, tuple):
        return tuple(_remap_output_value(item, path_map=path_map) for item in value)
    if isinstance(value, list):
        return [_remap_output_value(item, path_map=path_map) for item in value]
    raise ExecutionError(f"File output values must be path-like, got {type(value).__name__}")
