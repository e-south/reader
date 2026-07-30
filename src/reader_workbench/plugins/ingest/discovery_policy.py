"""Raw-file autodiscovery policy for ingest plugins."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Sequence
from pathlib import Path

from reader_workbench.errors import ParseError
from reader_workbench.workbench.input_discovery import (
    DEFAULT_INPUT_EXCLUDE as DEFAULT_EXCLUDE,
)
from reader_workbench.workbench.input_discovery import (
    DEFAULT_INPUT_ROOTS as DEFAULT_ROOTS,
)
from reader_workbench.workbench.input_discovery import (
    DEFAULT_WORKBOOK_INCLUDE as DEFAULT_INCLUDE,
)
from reader_workbench.workbench.paths import resolve_path_within_root


def _iter_candidates(root: Path, patterns: Sequence[str], recursive: bool) -> Iterable[Path]:
    if not root.exists():
        return []
    if recursive:
        for path in root.rglob("*"):
            if path.is_file() and any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
                yield path
        return
    for path in root.glob("*"):
        if path.is_file() and any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
            yield path


def discover_files(
    base: Path,
    *,
    roots: Sequence[str] | None = None,
    include: Sequence[str] = DEFAULT_INCLUDE,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    recursive: bool = False,
) -> list[Path]:
    roots = list(roots or DEFAULT_ROOTS)
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            root_path = resolve_path_within_root(root, root=base)
        except ValueError as err:
            raise ParseError(
                f"auto_roots entry {root!r} must stay under the experiment root after resolving symlinks."
            ) from err
        for path in _iter_candidates(root_path, include, recursive):
            try:
                confined_path = resolve_path_within_root(path, root=base)
            except ValueError as err:
                raise ParseError(
                    f"Discovered input {path!s} must stay under the experiment root after resolving symlinks."
                ) from err
            if any(fnmatch.fnmatch(path.name, pat) for pat in exclude):
                continue
            if confined_path in seen:
                continue
            out.append(confined_path)
            seen.add(confined_path)
    return sorted(out)
