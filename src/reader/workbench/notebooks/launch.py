from __future__ import annotations

import os
import signal
import socket
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from reader.errors import ConfigError

from ._launch_registry import (
    MarimoSessionRecord,
)
from ._launch_registry import (
    load_registry as _load_registry,
)
from ._launch_registry import (
    prune_registry as _prune_registry,
)
from ._launch_registry import (
    session_matches_current_inputs as _session_matches_current_inputs,
)
from ._launch_registry import (
    write_registry as _write_registry,
)
from ._launch_runtime import (
    MarimoRuntimePaths,
)
from ._launch_runtime import (
    build_env as _build_env,
)
from ._launch_runtime import (
    find_experiment_root as _find_experiment_root,
)
from ._launch_runtime import (
    find_repo_root as _find_repo_root,
)
from ._launch_runtime import (
    runtime_fingerprint as _runtime_fingerprint,
)
from ._launch_runtime import (
    runtime_paths_for_target as _runtime_paths_for_target,
)
from ._launch_runtime import (
    target_signature as _target_signature,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2718
DEFAULT_PORT_SCAN_LIMIT = 32


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


@dataclass(frozen=True)
class _LaunchContext:
    resolved_target: Path
    experiment_root: Path
    repo_root: Path
    notebook_mtime_ns: int
    notebook_size_bytes: int
    runtime_fingerprint: str


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


def _resolve_launch_context(target: Path) -> _LaunchContext:
    resolved_target = target.resolve()
    experiment_root = _find_experiment_root(resolved_target)
    repo_root = _find_repo_root(resolved_target)
    notebook_mtime_ns, notebook_size_bytes = _target_signature(resolved_target)
    return _LaunchContext(
        resolved_target=resolved_target,
        experiment_root=experiment_root,
        repo_root=repo_root,
        notebook_mtime_ns=notebook_mtime_ns,
        notebook_size_bytes=notebook_size_bytes,
        runtime_fingerprint=_runtime_fingerprint(repo_root),
    )


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
    context = _resolve_launch_context(target)
    runtime_paths = _runtime_paths_for_target(context.resolved_target)
    env = _build_env(context.repo_root, runtime_paths=runtime_paths, base_env=base_env)
    records = _prune_registry(_load_registry(runtime_paths.registry_path), pid_is_live=_pid_is_live)

    for record in records:
        if _session_matches_current_inputs(
            record,
            mode=mode,
            resolved_target=context.resolved_target,
            experiment_root=context.experiment_root,
            runtime_fingerprint=context.runtime_fingerprint,
            notebook_mtime_ns=context.notebook_mtime_ns,
            notebook_size_bytes=context.notebook_size_bytes,
            port_is_open=lambda host, port: _port_is_open(host, port),
        ):
            _write_registry(runtime_paths.registry_path, records)
            return MarimoLaunchPlan(
                cmd=(),
                env=env,
                url=f"http://{record.host}:{record.port}",
                port=record.port,
                host=record.host,
                target=context.resolved_target,
                runtime_paths=runtime_paths,
                reused_session=record,
            )

    terminated_sessions: list[MarimoSessionRecord] = []
    kept_records: list[MarimoSessionRecord] = []
    for record in records:
        if (
            record.mode == mode
            and record.experiment_root == str(context.experiment_root)
            and _terminate_pid(record.pid)
        ):
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
    cmd.append(str(context.resolved_target))

    _write_registry(runtime_paths.registry_path, kept_records)
    return MarimoLaunchPlan(
        cmd=tuple(cmd),
        env=env,
        url=f"http://{DEFAULT_HOST}:{port}",
        port=port,
        host=DEFAULT_HOST,
        target=context.resolved_target,
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
    records = _prune_registry(_load_registry(registry_path), pid_is_live=_pid_is_live)
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
    records = _prune_registry(_load_registry(registry_path), pid_is_live=_pid_is_live)
    _write_registry(registry_path, [record for record in records if record.pid != pid])


def open_url(url: str) -> None:
    try:
        webbrowser.open(url, new=0, autoraise=True)
    except Exception:
        return
