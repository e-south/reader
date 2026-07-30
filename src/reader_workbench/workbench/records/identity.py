from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from reader_workbench._version import package_version
from reader_workbench.errors import RecordError

_PACKAGED_SOURCE_SUFFIXES = frozenset({".py", ".pyi"})
_PACKAGED_DATA_GLOBS = ("workbench/notebooks/*.marimo.py.txt",)
_PACKAGED_DATA_NAMES = frozenset({"py.typed"})


def digest_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def is_sha256_digest(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


@dataclass(frozen=True)
class BuildIdentity:
    reader_version: str
    source_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.reader_version, str) or not self.reader_version.strip():
            raise RecordError("build_identity.reader_version must be a non-empty string")
        if not is_sha256_digest(self.source_digest):
            raise RecordError("build_identity.source_digest must be a sha256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "reader_version": self.reader_version,
            "source_digest": self.source_digest,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> BuildIdentity:
        if not isinstance(payload, dict) or set(payload) != {"reader_version", "source_digest"}:
            raise RecordError("build_identity must contain only reader_version and source_digest")
        return cls(
            reader_version=payload.get("reader_version"),
            source_digest=payload.get("source_digest"),
        )


def _packaged_runtime_files(package_root: Path) -> tuple[Path, ...]:
    files: set[Path] = set()
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root)
        if relative.parts[0] == "tests":
            continue
        if path.suffix in _PACKAGED_SOURCE_SUFFIXES or path.name in _PACKAGED_DATA_NAMES:
            files.add(path)
    for pattern in _PACKAGED_DATA_GLOBS:
        files.update(path for path in package_root.glob(pattern) if path.is_file())
    return tuple(sorted(files, key=lambda item: item.relative_to(package_root).as_posix()))


def _source_digest(package_root: Path) -> str:
    digest = hashlib.sha256()
    for path in _packaged_runtime_files(package_root):
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return "sha256:" + digest.hexdigest()


@cache
def current_build_identity() -> BuildIdentity:
    package_root = Path(__file__).resolve().parents[2]
    return BuildIdentity(reader_version=package_version(), source_digest=_source_digest(package_root))
