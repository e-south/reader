"""Shared atomic publication for response-window evidence bundles."""

from __future__ import annotations

import fcntl
import os
import shutil
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

PublishedBundle = TypeVar("PublishedBundle")


@dataclass(frozen=True)
class BundlePublication:
    """One staged bundle publication with explicit overwrite semantics."""

    destination: Path
    staging: Path
    backup: Path
    bundle_label: str
    overwrite: bool

    def publish(self, verify: Callable[[Path], PublishedBundle]) -> PublishedBundle:
        """Install and verify the staged bundle, restoring prior output on failure."""

        with _exclusive_parent_lock(self.destination.parent):
            _validate_destination(
                self.destination,
                bundle_label=self.bundle_label,
                overwrite=self.overwrite,
            )
            backup_created = False
            installed_identity: tuple[int, int] | None = None
            if self.destination.exists():
                self.destination.rename(self.backup)
                backup_created = True
            try:
                self.staging.rename(self.destination)
                installed_identity = _path_identity(self.destination)
                published = verify(self.destination)
            except BaseException as error:
                _rollback_publication(
                    publication=self,
                    backup_created=backup_created,
                    installed_identity=installed_identity,
                    error=error,
                )
                raise
            if backup_created:
                shutil.rmtree(self.backup)
            return published


@contextmanager
def bundle_publication(
    path: Path,
    *,
    bundle_label: str,
    overwrite: bool,
) -> Iterator[BundlePublication]:
    """Create a unique staging directory and clean it after publication."""

    destination = resolve_bundle_destination(
        path,
        bundle_label=bundle_label,
        overwrite=overwrite,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    publication = BundlePublication(
        destination=destination,
        staging=destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}",
        backup=destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}",
        bundle_label=bundle_label,
        overwrite=overwrite,
    )
    publication.staging.mkdir(parents=False)
    try:
        yield publication
    finally:
        if publication.staging.exists():
            shutil.rmtree(publication.staging)


def resolve_bundle_destination(
    path: Path,
    *,
    bundle_label: str,
    overwrite: bool,
) -> Path:
    """Resolve a bundle destination without following a final-path symlink."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError(f"{bundle_label} output must be a real directory path, not a symbolic link: {requested}")

    destination = requested.resolve()
    _validate_destination(destination, bundle_label=bundle_label, overwrite=overwrite)
    return destination


def _validate_destination(destination: Path, *, bundle_label: str, overwrite: bool) -> None:
    if destination.is_symlink():
        raise ValueError(f"{bundle_label} output must be a real directory path, not a symbolic link: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{bundle_label} output already exists: {destination}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"{bundle_label} output must be a real directory path, not a file: {destination}")


@contextmanager
def _exclusive_parent_lock(parent: Path) -> Iterator[None]:
    """Serialize Reader publishers without leaving a lock artifact behind."""

    descriptor = os.open(parent, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _rollback_publication(
    *,
    publication: BundlePublication,
    backup_created: bool,
    installed_identity: tuple[int, int] | None,
    error: BaseException,
) -> None:
    destination = publication.destination
    if installed_identity is not None and _path_identity(destination) == installed_identity:
        shutil.rmtree(destination)
    if not backup_created:
        return
    if not destination.exists() and not destination.is_symlink():
        publication.backup.rename(destination)
        return
    error.add_note(
        f"Previous {publication.bundle_label} output remains at {publication.backup}; "
        "the destination changed during rollback and was left untouched."
    )


__all__ = ["BundlePublication", "bundle_publication", "resolve_bundle_destination"]
