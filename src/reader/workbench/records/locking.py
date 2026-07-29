"""Filesystem-safe, process-shared locks for provenance publication."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from filelock import BaseFileLock, Timeout

if os.name == "posix":
    import fcntl
elif os.name == "nt":  # pragma: win32 cover
    import msvcrt


class ProvenanceFileLock(BaseFileLock):
    """A reentrant advisory lock that never truncates an existing target.

    ``BaseFileLock`` supplies timeout polling and per-thread reentrancy. This
    implementation owns the filesystem boundary: it opens without truncation,
    validates the opened descriptor, acquires an OS lock, and then proves that
    the locked inode is still the single-link object named by the lock path.
    """

    def _acquire(self) -> None:
        if os.name not in {"posix", "nt"}:  # pragma: no cover - fail closed on unsupported runtimes
            raise NotImplementedError("Secure provenance locks require POSIX flock or Windows msvcrt locking")

        descriptor = -1
        parent_descriptor = -1
        parent_stat: os.stat_result | None = None
        locked = False
        try:
            lock_path = Path(self.lock_file)
            if os.name == "posix":
                parent_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
                parent_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                parent_descriptor = os.open(lock_path.parent, parent_flags)
                parent_stat = os.fstat(parent_descriptor)
                if not stat.S_ISDIR(parent_stat.st_mode):
                    raise OSError("Provenance lock parent must be a directory")
                descriptor = os.open(
                    lock_path.name,
                    self._open_flags(),
                    self._context.mode,
                    dir_fd=parent_descriptor,
                )
                self._validate_descriptor(
                    descriptor,
                    lock_path=lock_path,
                    parent_descriptor=parent_descriptor,
                    parent_stat=parent_stat,
                )
            else:  # pragma: win32 cover
                descriptor = os.open(lock_path, self._open_flags(), self._context.mode)
                self._validate_descriptor(descriptor, lock_path=lock_path)

            if not self._try_lock(descriptor):
                return
            locked = True
            self._validate_descriptor(
                descriptor,
                lock_path=lock_path,
                parent_descriptor=parent_descriptor if parent_descriptor >= 0 else None,
                parent_stat=parent_stat,
            )
            self._context.lock_file_fd = descriptor
            descriptor = -1
            locked = False
        finally:
            if locked:
                self._unlock(descriptor)
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _release(self) -> None:
        descriptor = self._context.lock_file_fd
        if descriptor is None:  # pragma: no cover - BaseFileLock guards this path
            return
        self._context.lock_file_fd = None
        try:
            self._unlock(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _open_flags() -> int:
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_BINARY", 0)
        return flags

    @staticmethod
    def _validate_descriptor(
        descriptor: int,
        *,
        lock_path: Path,
        parent_descriptor: int | None = None,
        parent_stat: os.stat_result | None = None,
    ) -> None:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
            raise OSError("Provenance lock must be a regular file with a single link")

        if parent_descriptor is None:
            named_stat = os.stat(lock_path, follow_symlinks=False)
        else:
            named_stat = os.stat(lock_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            current_parent = os.stat(lock_path.parent, follow_symlinks=False)
            if parent_stat is None or not _same_inode(parent_stat, current_parent):
                raise OSError("Provenance lock parent changed while the lock was acquired")
        if not stat.S_ISREG(named_stat.st_mode) or named_stat.st_nlink != 1:
            raise OSError("Provenance lock must be a regular file with a single link")
        if not _same_inode(opened_stat, named_stat):
            raise OSError("Provenance lock target changed while the lock was acquired")

    @staticmethod
    def _try_lock(descriptor: int) -> bool:
        if os.name == "posix":
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    return False
                if exc.errno == errno.ENOSYS:
                    raise NotImplementedError("The filesystem does not support provenance file locks") from exc
                raise
            return True
        try:  # pragma: win32 cover
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:  # pragma: win32 cover
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    @staticmethod
    def _unlock(descriptor: int) -> None:
        if os.name == "posix":
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        else:  # pragma: win32 cover
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


@contextmanager
def provenance_lock_scope(
    lock: Any,
    *,
    acquire_error: Exception,
    release_error: Exception,
    release_note: str,
) -> Iterator[None]:
    """Hold a provenance lock without masking a protected operation failure."""

    try:
        lock.acquire()
    except (Timeout, OSError, NotImplementedError) as exc:
        raise acquire_error from exc
    body_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        try:
            lock.release()
        except (Timeout, OSError, NotImplementedError) as exc:
            if body_error is not None:
                body_error.add_note(f"{release_note} ({type(exc).__name__}).")
            else:
                raise release_error from exc


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)
