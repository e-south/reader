from __future__ import annotations

from importlib import import_module
from pathlib import Path

from reader.errors import ConfigError
from reader.workbench.paths import resolve_confined_sink_root, resolve_path_within_root


def write_experiment_notebook(
    target: Path,
    *,
    experiment_root: Path,
    notebooks_root: Path,
    template: str,
    overwrite: bool = False,
    plot_specs: list[dict] | None = None,
) -> tuple[Path, bool]:
    target = _confined_notebook_target(
        target,
        experiment_root=experiment_root,
        notebooks_root=notebooks_root,
    )
    if target.exists() and not overwrite:
        return target, False
    descriptor = import_module("reader.workbench.templates").resolve_notebook_template_descriptor(template)
    body = descriptor.load_body()
    if descriptor.capabilities.inject_plot_specs and "__PLOT_SPECS__" in body:
        payload = plot_specs or []
        body = body.replace("__PLOT_SPECS__", repr(payload))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target, True


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
