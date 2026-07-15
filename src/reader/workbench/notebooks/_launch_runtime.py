from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from reader.errors import ConfigError

RUNTIME_FINGERPRINT_SUFFIXES = (".py", ".txt")


@dataclass(frozen=True)
class MarimoRuntimePaths:
    root: Path
    registry_path: Path
    xdg_config_home: Path
    xdg_state_home: Path
    xdg_cache_home: Path
    mplconfigdir: Path


def find_repo_root(start: Path) -> Path:
    for base in [start.resolve()] + list(start.resolve().parents):
        if (base / "pyproject.toml").exists():
            return base
    raise ConfigError(f"Could not find repository root from {start}")


def resolve_repo_root(target: Path, *, repo_root: Path | None = None) -> Path:
    if repo_root is None:
        return find_repo_root(target)
    resolved_root = repo_root.expanduser().resolve()
    if not (resolved_root / "pyproject.toml").is_file():
        raise ConfigError(f"Reader repository root does not contain pyproject.toml: {resolved_root}")
    return resolved_root


def find_experiment_root(start: Path) -> Path:
    for base in [start.resolve()] + list(start.resolve().parents):
        if (base / "config.yaml").exists():
            return base
    return start.resolve().parent


def runtime_paths_for_repo_root(repo_root: Path) -> MarimoRuntimePaths:
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


def target_signature(target: Path) -> tuple[int, int]:
    stat = target.resolve().stat()
    return stat.st_mtime_ns, stat.st_size


def runtime_fingerprint(repo_root: Path) -> str:
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


def build_env(
    repo_root: Path,
    *,
    runtime_paths: MarimoRuntimePaths,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    pythonpath_parts = [str(repo_root)]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["XDG_CONFIG_HOME"] = str(runtime_paths.xdg_config_home)
    env["XDG_STATE_HOME"] = str(runtime_paths.xdg_state_home)
    env["XDG_CACHE_HOME"] = str(runtime_paths.xdg_cache_home)
    env["READER_MPLCONFIGDIR"] = str(runtime_paths.mplconfigdir)
    env["MPLCONFIGDIR"] = str(runtime_paths.mplconfigdir)
    return env
