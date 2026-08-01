"""Filesystem transaction for starting a fresh generated-output epoch."""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Iterable
from pathlib import Path
from uuid import uuid4

from reader_workbench.errors import RecordError
from reader_workbench.workbench.paths import resolve_confined_sink_root


def replace_generated_epoch(
    *,
    outputs_root: Path,
    artifacts_root: Path,
    manifests_root: Path,
    plots_root: Path,
    exports_root: Path,
    log_path: Path,
    preserved_paths: Iterable[Path],
    lock_path: Path,
    initialize: Callable[[], str],
) -> str:
    """Stage owned output roots, initialize a new epoch, and remove the old epoch.

    The writer lock remains in place while the other manifest entries move. Paths
    outside the named owned roots are never enumerated or removed.
    """

    root, manifests, lock, owned_roots, owned_files = validate_generated_epoch_boundary(
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
        manifests_root=manifests_root,
        plots_root=plots_root,
        exports_root=exports_root,
        log_path=log_path,
        preserved_paths=preserved_paths,
        lock_path=lock_path,
    )

    staging = root / f".reader-reset.{uuid4()}.staging"
    staged_roots: list[tuple[Path, Path]] = []
    staged_manifest_entries: list[tuple[Path, Path]] = []
    staged_files: list[tuple[Path, Path]] = []
    initialization_started = False
    cleanup_started = False
    try:
        staging.mkdir(mode=0o700)
        roots_staging = staging / "roots"
        manifests_staging = staging / "manifests"
        files_staging = staging / "files"
        roots_staging.mkdir()
        manifests_staging.mkdir()
        files_staging.mkdir()

        for index, source in enumerate(owned_roots):
            if not source.exists():
                continue
            destination = roots_staging / f"{index}-{source.name}"
            source.rename(destination)
            staged_roots.append((source, destination))

        if manifests.exists():
            for source in sorted(manifests.iterdir(), key=lambda path: path.name):
                if source == lock:
                    continue
                destination = manifests_staging / source.name
                source.rename(destination)
                staged_manifest_entries.append((source, destination))

        for index, source in enumerate(owned_files):
            if not source.exists():
                continue
            destination = files_staging / f"{index}-{source.name}"
            source.rename(destination)
            staged_files.append((source, destination))

        initialization_started = True
        epoch_id = initialize()
        cleanup_started = True
        shutil.rmtree(staging)
        return epoch_id
    except BaseException as exc:
        if cleanup_started:
            raise RecordError(
                "The fresh generated-output epoch was initialized, but Reader could not remove all staged prior "
                f"state at {staging}. Do not retry blindly; inspect the staging directory and run reader verify."
            ) from exc
        if staging.exists() and not cleanup_started:
            rollback_succeeded = False
            try:
                _rollback(
                    manifests=manifests,
                    lock=lock,
                    owned_roots=owned_roots,
                    owned_files=owned_files,
                    staged_roots=staged_roots,
                    staged_manifest_entries=staged_manifest_entries,
                    staged_files=staged_files,
                    discard_current=initialization_started,
                )
            except BaseException as rollback_error:
                exc.add_note(
                    "Reader also could not fully restore the prior generated-output epoch "
                    f"({rollback_error}). Remaining staged state is retained at {staging}."
                )
            else:
                rollback_succeeded = True
            if rollback_succeeded:
                try:
                    shutil.rmtree(staging)
                except OSError as cleanup_error:
                    exc.add_note(f"Reader restored the prior epoch but could not remove empty staging at {staging}.")
                    exc.add_note(f"Staging cleanup failed with {type(cleanup_error).__name__}.")
        if isinstance(exc, RecordError):
            raise
        raise RecordError(f"Could not start a fresh generated-output epoch: {exc}") from exc


def validate_generated_epoch_boundary(
    *,
    outputs_root: Path,
    artifacts_root: Path,
    manifests_root: Path,
    plots_root: Path,
    exports_root: Path,
    log_path: Path,
    preserved_paths: Iterable[Path],
    lock_path: Path,
) -> tuple[Path, Path, Path, tuple[Path, ...], tuple[Path, ...]]:
    """Validate the complete reset boundary without mutating it."""

    root = Path(outputs_root).expanduser().absolute()
    manifests = Path(manifests_root).expanduser().absolute()
    lock = Path(lock_path).expanduser().absolute()
    owned_files = (_confined(log_path, root=root, label="reader log"),)
    owned_roots = _minimal_owned_roots(
        root=root,
        candidates=(artifacts_root, plots_root, exports_root),
        manifests_root=manifests,
    )
    preserved = tuple(_confined(path, root=root, label="preserved output") for path in preserved_paths)
    assert_no_interrupted_epoch(root)
    _validate_reset_boundary(
        root=root,
        manifests=manifests,
        lock=lock,
        owned_roots=owned_roots,
        preserved=preserved,
    )
    for owned_file in owned_files:
        _validate_existing_file(owned_file)
    return root, manifests, lock, owned_roots, owned_files


def assert_no_interrupted_epoch(outputs_root: Path) -> None:
    """Reject writes while recovery evidence from an interrupted reset remains."""

    root = Path(outputs_root).expanduser().absolute()
    stale_transactions = _stale_transactions(root)
    if not stale_transactions:
        return
    paths = ", ".join(str(path) for path in stale_transactions)
    raise RecordError(
        "An unfinished generated-output reset exists at "
        f"{paths}. Reader will not mutate outputs or discard recovery evidence. "
        "Inspect the retained roots/, manifests/, and files/ entries; restore the prior epoch or archive "
        "confirmed residue, then remove the staging directory before running Reader again."
    )


def _minimal_owned_roots(
    *,
    root: Path,
    candidates: Iterable[Path],
    manifests_root: Path,
) -> tuple[Path, ...]:
    normalized: list[Path] = []
    for candidate in candidates:
        path = _confined(candidate, root=root, label="generated output")
        if path == root:
            raise RecordError(
                "A fresh generated-output epoch requires plots and exports to use dedicated subdirectories; "
                "a flattened sink would make generated files indistinguishable from preserved output-root content."
            )
        if _contains(manifests_root, path):
            continue
        if path not in normalized:
            normalized.append(path)
    return tuple(
        path
        for path in sorted(normalized, key=lambda item: len(item.parts))
        if not any(_contains(parent, path) for parent in normalized if parent != path)
    )


def _validate_reset_boundary(
    *,
    root: Path,
    manifests: Path,
    lock: Path,
    owned_roots: tuple[Path, ...],
    preserved: tuple[Path, ...],
) -> None:
    _confined(manifests, root=root, label="manifests")
    _confined(lock, root=manifests, label="record lock")
    for owned in (*owned_roots, manifests):
        if any(keep != root and _overlaps(owned, keep) for keep in preserved):
            raise RecordError(
                f"Generated output sink {owned} overlaps preserved output path; "
                "use distinct plots, exports, and notebooks subdirectories before resetting records."
            )
        _validate_existing_tree(owned)


def _validate_existing_tree(root: Path) -> None:
    if root.is_symlink():
        raise RecordError(f"Generated output sink must not be a symlink: {root}")
    if not root.exists():
        return
    if not root.is_dir():
        raise RecordError(f"Generated output sink must be a directory: {root}")
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            candidate = current_path / name
            if candidate.is_symlink():
                raise RecordError(f"Generated output state must not contain symlinks: {candidate}")


def _validate_existing_file(path: Path) -> None:
    if path.is_symlink():
        raise RecordError(f"Reader-owned output must not be a symlink: {path}")
    if not path.exists():
        return
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RecordError(f"Reader-owned output must be a regular file with a single link: {path}")


def _stale_transactions(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.name.startswith(".reader-reset.") and path.name.endswith(".staging")
            ),
            key=lambda path: path.name,
        )
    )


def _rollback(
    *,
    manifests: Path,
    lock: Path,
    owned_roots: tuple[Path, ...],
    owned_files: tuple[Path, ...],
    staged_roots: list[tuple[Path, Path]],
    staged_manifest_entries: list[tuple[Path, Path]],
    staged_files: list[tuple[Path, Path]],
    discard_current: bool,
) -> None:
    if discard_current and manifests.exists():
        for current in tuple(manifests.iterdir()):
            if current != lock:
                _remove_entry(current)
    elif not manifests.exists():
        manifests.mkdir(parents=True)
    for source, destination in reversed(staged_manifest_entries):
        if source.exists() or source.is_symlink():
            _remove_entry(source)
        destination.rename(source)

    if discard_current:
        for source in owned_files:
            if source.exists() or source.is_symlink():
                _remove_entry(source)
    for source, destination in reversed(staged_files):
        if source.exists() or source.is_symlink():
            if not discard_current:
                raise RecordError(f"Cannot restore staged generated output because its target reappeared: {source}")
            _remove_entry(source)
        destination.rename(source)

    if discard_current:
        for source in reversed(owned_roots):
            if source.exists() or source.is_symlink():
                _remove_entry(source)
    for source, destination in reversed(staged_roots):
        if source.exists() or source.is_symlink():
            if not discard_current:
                raise RecordError(f"Cannot restore staged generated output because its target reappeared: {source}")
            _remove_entry(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(source)


def _remove_entry(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _confined(path: Path, *, root: Path, label: str) -> Path:
    try:
        return resolve_confined_sink_root(path, root=root, label=label)
    except ValueError as exc:
        raise RecordError(str(exc)) from exc


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _overlaps(first: Path, second: Path) -> bool:
    return _contains(first, second) or _contains(second, first)
