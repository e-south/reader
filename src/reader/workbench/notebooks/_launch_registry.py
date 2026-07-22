from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarimoSessionRecord:
    pid: int
    port: int
    host: str
    mode: str
    notebook: str
    experiment_root: str
    repo_root: str
    launched_at: float
    notebook_mtime_ns: int | None = None
    notebook_size_bytes: int | None = None
    runtime_fingerprint: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MarimoSessionRecord | None:
        try:
            return cls(
                pid=int(payload["pid"]),
                port=int(payload["port"]),
                host=str(payload["host"]),
                mode=str(payload["mode"]),
                notebook=str(payload["notebook"]),
                experiment_root=str(payload["experiment_root"]),
                repo_root=str(payload["repo_root"]),
                launched_at=float(payload["launched_at"]),
                notebook_mtime_ns=(
                    int(payload["notebook_mtime_ns"]) if payload.get("notebook_mtime_ns") is not None else None
                ),
                notebook_size_bytes=(
                    int(payload["notebook_size_bytes"]) if payload.get("notebook_size_bytes") is not None else None
                ),
                runtime_fingerprint=(
                    str(payload["runtime_fingerprint"]) if payload.get("runtime_fingerprint") is not None else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None


def load_registry(registry_path: Path) -> list[MarimoSessionRecord]:
    if not registry_path.exists():
        return []
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    records: list[MarimoSessionRecord] = []
    for item in payload:
        if isinstance(item, dict):
            record = MarimoSessionRecord.from_dict(item)
            if record is not None:
                records.append(record)
    return records


def write_registry(registry_path: Path, records: list[MarimoSessionRecord]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prune_registry(
    records: list[MarimoSessionRecord],
    *,
    pid_is_live: Callable[[int], bool],
) -> list[MarimoSessionRecord]:
    pruned: list[MarimoSessionRecord] = []
    for record in records:
        notebook_path = Path(record.notebook)
        if not notebook_path.exists():
            continue
        if not pid_is_live(record.pid):
            continue
        pruned.append(record)
    return pruned


def session_matches_current_inputs(
    record: MarimoSessionRecord,
    *,
    mode: str,
    resolved_target: Path,
    experiment_root: Path,
    runtime_fingerprint: str,
    notebook_mtime_ns: int,
    notebook_size_bytes: int,
    port_is_open: Callable[[str, int], bool],
) -> bool:
    if record.mode != mode:
        return False
    if record.notebook != str(resolved_target):
        return False
    if record.experiment_root != str(experiment_root):
        return False
    if record.notebook_mtime_ns != notebook_mtime_ns:
        return False
    if record.notebook_size_bytes != notebook_size_bytes:
        return False
    if record.runtime_fingerprint != runtime_fingerprint:
        return False
    return port_is_open(record.host, record.port)
