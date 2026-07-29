from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from reader.errors import RecordError


def digest_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class BuildIdentity:
    reader_version: str
    source_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.reader_version, str) or not self.reader_version.strip():
            raise RecordError("build_identity.reader_version must be a non-empty string")
        if not isinstance(self.source_digest, str) or not self.source_digest.startswith("sha256:"):
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


@cache
def current_build_identity() -> BuildIdentity:
    try:
        reader_version = importlib.metadata.version("reader")
    except importlib.metadata.PackageNotFoundError:
        reader_version = "0+uninstalled"

    package_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    source_files = (
        path
        for path in package_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    )
    for path in sorted(source_files, key=lambda item: str(item.relative_to(package_root))):
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return BuildIdentity(reader_version=reader_version, source_digest="sha256:" + digest.hexdigest())
