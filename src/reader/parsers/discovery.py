"""
--------------------------------------------------------------------------------
<reader project>
src/reader/parsers/discovery.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_ROOTS = ("./inputs",)
DEFAULT_INCLUDE = ("*.xlsx", "*.xls")
DEFAULT_EXCLUDE = ("~$*", "._*", "#*#", "*.tmp", "*.temp", "*.bak")


def _iter_candidates(root: Path, patterns: Sequence[str], recursive: bool) -> Iterable[Path]:
    if not root.exists():
        return []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
                yield p
    else:
        for p in root.glob("*"):
            if p.is_file() and any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
                yield p


def discover_files(
    base: Path,
    *,
    roots: Sequence[str] | None = None,
    include: Sequence[str] = DEFAULT_INCLUDE,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    recursive: bool = False,
) -> list[Path]:
    """
    Return candidate raw files under the given roots (relative to `base` = experiment dir).
    - roots: directories to search (default: ./inputs)
    - include: filename patterns to include
    - exclude: filename patterns to exclude
    - recursive: whether to search subdirectories
    """
    roots = list(roots or DEFAULT_ROOTS)
    out: list[Path] = []
    seen: set[Path] = set()

    for r in roots:
        root_path = (base / r).resolve()
        for p in _iter_candidates(root_path, include, recursive):
            name = p.name
            if any(fnmatch.fnmatch(name, pat) for pat in exclude):
                continue
            if p not in seen:
                out.append(p)
                seen.add(p)
    return sorted(out)
