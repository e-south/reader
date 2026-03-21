from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import sys
import time
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path

from reader.errors import ConfigError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2718
DEFAULT_PORT_SCAN_LIMIT = 32
RUNTIME_FINGERPRINT_SUFFIXES = (".py", ".txt")


@dataclass(frozen=True)
class MarimoRuntimePaths:
    root: Path
    registry_path: Path
    xdg_config_home: Path
    xdg_state_home: Path
    xdg_cache_home: Path
    mplconfigdir: Path


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


@dataclass(frozen=True)
class MarimoLaunchPlan:
    cmd: tuple[str, ...]
    env: dict[str, str]
    url: str
    port: int
    host: str
    target: Path
    runtime_paths: MarimoRuntimePaths
    reused_session: MarimoSessionRecord | None = None
    terminated_sessions: tuple[MarimoSessionRecord, ...] = ()


def _find_repo_root(start: Path) -> Path:
    for base in [start.resolve()] + list(start.resolve().parents):
        if (base / "pyproject.toml").exists():
            return base
    raise ConfigError(f"Could not find repository root from {start}")


def _find_experiment_root(start: Path) -> Path:
    for base in [start.resolve()] + list(start.resolve().parents):
        if (base / "config.yaml").exists():
            return base
    return start.resolve().parent


def _runtime_paths_for_target(target: Path) -> MarimoRuntimePaths:
    repo_root = _find_repo_root(target)
    root = repo_root / ".cache" / "marimo"
    xdg_config_home = root / "xdg-config"
    xdg_state_home = root / "xdg-state"
    xdg_cache_home = root / "xdg-cache"
    mplconfigdir = root / "matplotlib"
    for path in (root, xdg_config_home, xdg_state_home, xdg_cache_home, mplconfigdir):
        path.mkdir(parents=True, exist_ok=True)
    return MarimoRuntimePaths(
        root=root,
        registry_path=root / "sessions.json",
        xdg_config_home=xdg_config_home,
        xdg_state_home=xdg_state_home,
        xdg_cache_home=xdg_cache_home,
        mplconfigdir=mplconfigdir,
    )


def _load_registry(registry_path: Path) -> list[MarimoSessionRecord]:
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


def _write_registry(registry_path: Path, records: list[MarimoSessionRecord]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _target_signature(target: Path) -> tuple[int, int]:
    stat = target.resolve().stat()
    return stat.st_mtime_ns, stat.st_size


def _runtime_fingerprint(repo_root: Path) -> str:
    resolved_root = repo_root.resolve()
    hasher = hashlib.sha256()
    candidates: list[Path] = []
    pyproject = resolved_root / "pyproject.toml"
    if pyproject.exists():
        candidates.append(pyproject)
    source_root = resolved_root / "src" / "reader"
    if source_root.exists():
        for suffix in RUNTIME_FINGERPRINT_SUFFIXES:
            candidates.extend(path for path in source_root.rglob(f"*{suffix}") if path.is_file())
    for path in sorted({item.resolve() for item in candidates}):
        stat = path.stat()
        relative = path.relative_to(resolved_root)
        hasher.update(f"{relative}:{stat.st_mtime_ns}:{stat.st_size}\n".encode())
    return hasher.hexdigest()


def _pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _port_is_open(host: str, port: int, *, timeout: float = 0.15) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _terminate_pid(pid: int, *, grace_seconds: float = 1.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _pid_is_live(pid):
            return True
        time.sleep(0.05)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return not _pid_is_live(pid)


def _prune_registry(records: list[MarimoSessionRecord]) -> list[MarimoSessionRecord]:
    pruned: list[MarimoSessionRecord] = []
    for record in records:
        notebook_path = Path(record.notebook)
        if not notebook_path.exists():
            continue
        if not _pid_is_live(record.pid):
            continue
        pruned.append(record)
    return pruned


def _session_matches_current_inputs(
    record: MarimoSessionRecord,
    *,
    mode: str,
    resolved_target: Path,
    experiment_root: Path,
    runtime_fingerprint: str,
    notebook_mtime_ns: int,
    notebook_size_bytes: int,
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
    return _port_is_open(record.host, record.port)


def _build_env(target: Path, *, base_env: dict[str, str] | None = None) -> tuple[dict[str, str], MarimoRuntimePaths]:
    env = dict(base_env or os.environ)
    runtime_paths = _runtime_paths_for_target(target)
    repo_root = _find_repo_root(target)
    pythonpath_parts = [str(repo_root)]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["READER_MARIMO_RUNTIME_PATCH"] = "1"
    env["XDG_CONFIG_HOME"] = str(runtime_paths.xdg_config_home)
    env["XDG_STATE_HOME"] = str(runtime_paths.xdg_state_home)
    env["XDG_CACHE_HOME"] = str(runtime_paths.xdg_cache_home)
    env["READER_MPLCONFIGDIR"] = str(runtime_paths.mplconfigdir)
    env["MPLCONFIGDIR"] = str(runtime_paths.mplconfigdir)
    return env, runtime_paths


def _choose_port(
    *,
    host: str,
    preferred_port: int | None,
    records: list[MarimoSessionRecord],
    scan_limit: int = DEFAULT_PORT_SCAN_LIMIT,
) -> int:
    occupied_ports = {record.port for record in records if _pid_is_live(record.pid)}

    if preferred_port is not None:
        if preferred_port in occupied_ports or _port_is_open(host, preferred_port):
            raise ConfigError(f"Port {preferred_port} is already in use on {host}")
        return preferred_port

    for port in range(DEFAULT_PORT, DEFAULT_PORT + scan_limit):
        if port in occupied_ports:
            continue
        if not _port_is_open(host, port):
            return port
    raise ConfigError(
        f"No free Marimo port found on {host} in range {DEFAULT_PORT}-{DEFAULT_PORT + scan_limit - 1}. "
        "Close existing sessions or provide --port."
    )


def plan_marimo_launch(
    *,
    mode: str,
    target: Path,
    headless: bool = False,
    preferred_port: int | None = None,
    base_env: dict[str, str] | None = None,
) -> MarimoLaunchPlan:
    resolved_target = target.resolve()
    experiment_root = _find_experiment_root(resolved_target)
    repo_root = _find_repo_root(resolved_target)
    env, runtime_paths = _build_env(resolved_target, base_env=base_env)
    records = _prune_registry(_load_registry(runtime_paths.registry_path))
    notebook_mtime_ns, notebook_size_bytes = _target_signature(resolved_target)
    runtime_fingerprint = _runtime_fingerprint(repo_root)

    for record in records:
        if _session_matches_current_inputs(
            record,
            mode=mode,
            resolved_target=resolved_target,
            experiment_root=experiment_root,
            runtime_fingerprint=runtime_fingerprint,
            notebook_mtime_ns=notebook_mtime_ns,
            notebook_size_bytes=notebook_size_bytes,
        ):
            _write_registry(runtime_paths.registry_path, records)
            return MarimoLaunchPlan(
                cmd=(),
                env=env,
                url=f"http://{record.host}:{record.port}",
                port=record.port,
                host=record.host,
                target=resolved_target,
                runtime_paths=runtime_paths,
                reused_session=record,
            )

    terminated_sessions: list[MarimoSessionRecord] = []
    kept_records: list[MarimoSessionRecord] = []
    for record in records:
        if record.mode == mode and record.experiment_root == str(experiment_root) and _terminate_pid(record.pid):
            terminated_sessions.append(record)
            continue
        kept_records.append(record)

    port = _choose_port(host=DEFAULT_HOST, preferred_port=preferred_port, records=kept_records)

    cmd = [sys.executable, "-m", "marimo", mode, "--host", DEFAULT_HOST, "--port", str(port)]
    if mode == "edit":
        cmd.append("--skip-update-check")
    if headless:
        cmd.append("--headless")
        cmd.append("--no-token")
    cmd.append(str(resolved_target))

    _write_registry(runtime_paths.registry_path, kept_records)
    return MarimoLaunchPlan(
        cmd=tuple(cmd),
        env=env,
        url=f"http://{DEFAULT_HOST}:{port}",
        port=port,
        host=DEFAULT_HOST,
        target=resolved_target,
        runtime_paths=runtime_paths,
        terminated_sessions=tuple(terminated_sessions),
    )


def register_managed_session(
    *,
    registry_path: Path,
    pid: int,
    port: int,
    host: str,
    mode: str,
    target: Path,
) -> None:
    records = _prune_registry(_load_registry(registry_path))
    target_mtime_ns, target_size_bytes = _target_signature(target)
    runtime_fingerprint = _runtime_fingerprint(_find_repo_root(target))
    record = MarimoSessionRecord(
        pid=pid,
        port=port,
        host=host,
        mode=mode,
        notebook=str(target.resolve()),
        experiment_root=str(_find_experiment_root(target)),
        repo_root=str(_find_repo_root(target)),
        launched_at=time.time(),
        notebook_mtime_ns=target_mtime_ns,
        notebook_size_bytes=target_size_bytes,
        runtime_fingerprint=runtime_fingerprint,
    )
    records = [existing for existing in records if existing.pid != pid]
    records.append(record)
    _write_registry(registry_path, records)


def unregister_managed_session(*, registry_path: Path, pid: int) -> None:
    records = _prune_registry(_load_registry(registry_path))
    _write_registry(registry_path, [record for record in records if record.pid != pid])


def open_url(url: str) -> None:
    try:
        webbrowser.open(url, new=0, autoraise=True)
    except Exception:
        return
