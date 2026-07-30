from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reader.workbench.graph import resolve_workbench
from reader.workbench.notebooks.components import (
    NotebookDeliverables,
    NotebookOverview,
    build_notebook_deliverable_selector,
    build_notebook_overview,
    collect_notebook_deliverables,
    render_notebook_deliverable_viewport,
    render_notebook_overview_panel,
)

from .facade import open_experiment
from .models import Experiment


@dataclass(frozen=True)
class NotebookContext:
    """Stable experiment context used by Reader-generated notebooks."""

    experiment: Experiment
    experiment_root: Path
    outputs_dir: Path
    notebooks_dir: Path
    pipeline_step_ids: tuple[str, ...]


def load_notebook_context(start: str | Path) -> NotebookContext:
    """Open the owning experiment for a generated notebook path."""

    experiment = open_experiment(start)
    decl = experiment._declaration
    layout = decl.experiment_semantics.layout
    notebooks_dir = (
        layout.outputs_dir
        if layout.notebooks_subdir in {"", ".", "./"}
        else layout.outputs_dir / layout.notebooks_subdir
    )
    workbench = resolve_workbench(decl)
    return NotebookContext(
        experiment=experiment,
        experiment_root=decl.experiment.root,
        outputs_dir=layout.outputs_dir,
        notebooks_dir=notebooks_dir,
        pipeline_step_ids=tuple(step.id for step in workbench.pipeline),
    )


__all__ = [
    "NotebookContext",
    "NotebookDeliverables",
    "NotebookOverview",
    "build_notebook_deliverable_selector",
    "build_notebook_overview",
    "collect_notebook_deliverables",
    "load_notebook_context",
    "render_notebook_deliverable_viewport",
    "render_notebook_overview_panel",
]
