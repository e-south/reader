from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from reader_workbench.errors import ConfigError
from reader_workbench.workbench.paths import resolve_confined_sink_root, resolve_path_within_root

CANONICAL_NOTEBOOK_ID = "notebook/eda"
_CANONICAL_NOTEBOOK_SOURCE = "eda.marimo.py.txt"


def write_experiment_notebook(
    target: Path,
    *,
    experiment_root: Path,
    notebooks_root: Path,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    target = _confined_notebook_target(
        target,
        experiment_root=experiment_root,
        notebooks_root=notebooks_root,
    )
    if target.exists() and not overwrite:
        return target, False
    body = _load_canonical_notebook_body()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target, True


def _load_canonical_notebook_body() -> str:
    try:
        return files(__package__).joinpath(_CANONICAL_NOTEBOOK_SOURCE).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise ConfigError(f"Canonical notebook source {CANONICAL_NOTEBOOK_ID!r} is unavailable.") from exc


def _confined_notebook_target(target: Path, *, experiment_root: Path, notebooks_root: Path) -> Path:
    try:
        owned_root = resolve_confined_sink_root(
            notebooks_root,
            root=experiment_root,
            label="notebooks",
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    candidate = Path(target).expanduser()
    if not candidate.is_absolute():
        candidate = owned_root / candidate
    candidate = candidate.absolute()
    if candidate.is_symlink():
        raise ConfigError(f"Notebook target must not be a symlink: {candidate}")
    try:
        resolve_confined_sink_root(candidate.parent, root=owned_root, label="notebook target parent")
        return resolve_path_within_root(candidate, root=owned_root)
    except ValueError as exc:
        raise ConfigError(f"Notebook target must stay within the configured notebooks root: {candidate}") from exc
