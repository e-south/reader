from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from reader.workbench import resolve_workbench
from reader.workbench.decl import load_workbench_decl

if TYPE_CHECKING:
    from reader.workbench.decl import WorkbenchDecl
    from reader.workbench.model import Workbench


@dataclass(frozen=True)
class NotebookWorkbenchContext:
    config_path: Path
    experiment_root: Path
    decl: WorkbenchDecl
    workbench: Workbench
    outputs_dir: Path
    plots_dir: Path
    exports_dir: Path


def find_notebook_experiment_root(start: Path) -> Path:
    for base in [start] + list(start.parents):
        if (base / "config.yaml").exists():
            return base
    raise RuntimeError(
        "No config.yaml found. Place this notebook under an experiment directory or set exp_dir manually."
    )


def load_notebook_workbench_context(start: Path) -> NotebookWorkbenchContext:
    experiment_root = find_notebook_experiment_root(start)
    config_path = experiment_root / "config.yaml"
    try:
        runtime_module = import_module("reader.runtime")
        protocols = runtime_module.builtin_runtime().protocols
        decl = load_workbench_decl(config_path, protocols=protocols)
    except Exception as exc:
        raise RuntimeError(f"Failed to read config.yaml: {exc}") from exc
    workbench = resolve_workbench(decl)
    outputs_dir = decl.experiment_semantics.layout.outputs_dir.resolve()
    return NotebookWorkbenchContext(
        config_path=config_path,
        experiment_root=experiment_root,
        decl=decl,
        workbench=workbench,
        outputs_dir=outputs_dir,
        plots_dir=(outputs_dir / "plots").resolve(),
        exports_dir=(outputs_dir / "exports").resolve(),
    )
