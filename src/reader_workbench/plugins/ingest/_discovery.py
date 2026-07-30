from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from reader_workbench.errors import ParseError

from .discovery_policy import DEFAULT_ROOTS, discover_files


def auto_pick_discovered_file(
    files: Sequence[Path],
    mode: str,
    *,
    singular_label: str,
) -> Path:
    if mode == "single":
        if len(files) != 1:
            raise ParseError(
                f"Auto-discovery expected exactly one {singular_label}, found "
                f"{len(files)}:\n- " + "\n- ".join(str(path) for path in files)
            )
        return files[0]
    if mode == "latest":
        return max(files, key=lambda path: path.stat().st_mtime)
    raise ParseError(f"Unknown auto_pick mode {mode!r} (expected: single|latest|merge)")


def discover_auto_input_files(
    *,
    exp_dir: Path,
    auto_roots: Sequence[str] | None,
    auto_include: Sequence[str],
    auto_exclude: Sequence[str],
    auto_recursive: bool,
    auto_pick: str,
    discovery_label: str,
    singular_label: str,
) -> list[Path]:
    roots = list(auto_roots or DEFAULT_ROOTS)
    files = discover_files(
        exp_dir,
        roots=roots,
        include=list(auto_include),
        exclude=list(auto_exclude),
        recursive=auto_recursive,
    )
    if not files:
        raise ParseError(
            f"No {discovery_label} discovered under {roots} "
            f"(include={list(auto_include)}, exclude={list(auto_exclude)}).\n"
            "Hint: put raw files under ./inputs (default), or set auto_roots / reads.raw explicitly."
        )
    if auto_pick in {"single", "latest"}:
        return [auto_pick_discovered_file(files, auto_pick, singular_label=singular_label)]
    if auto_pick == "merge":
        return files
    raise ParseError(f"Unknown auto_pick mode {auto_pick!r} (expected: single|latest|merge)")
