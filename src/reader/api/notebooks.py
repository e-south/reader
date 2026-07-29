from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from reader.errors import ConfigError
from reader.workbench.graph import resolve_workbench
from reader.workbench.notebooks.components import (
    NotebookDeliverables,
    NotebookOverview,
    build_dataframe_record_catalog,
    build_design_treatment_summary_rows,
    build_notebook_deliverable_selector,
    build_notebook_overview,
    collect_notebook_deliverables,
    render_notebook_deliverable_viewport,
    render_notebook_overview_panel,
    select_default_dataframe_record,
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
    pipeline_steps: tuple[CompiledStep, ...]
    protocol_inputs: Mapping[str, Any]
    ordered_state_spaces: Mapping[str, OrderedStateSpaceContext]


@dataclass(frozen=True)
class CompiledStep:
    """Domain-neutral metadata for one compiled pipeline step."""

    step_id: str
    plugin_id: str
    domain: str
    family: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CompiledStepConfig:
    """A compiled pipeline step with its effective protocol-bound configuration."""

    step: CompiledStep
    values: Mapping[str, Any]


@dataclass(frozen=True)
class OrderedStateSpaceContext:
    """Public notebook projection of one configured ordered state space."""

    ref: str
    column: str
    state_ids: tuple[str, ...]
    source_values: Mapping[str, str]
    case_sensitive: bool


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
    pipeline_steps = tuple(
        CompiledStep(
            step_id=step.id,
            plugin_id=step.plugin,
            domain=descriptor.domain,
            family=descriptor.family,
            tags=tuple(descriptor.tags),
        )
        for step in workbench.pipeline
        for descriptor in (experiment._runtime.plugins.resolve_descriptor(step.plugin),)
    )
    ordered_state_spaces = {
        ref: OrderedStateSpaceContext(
            ref=resolved.ref,
            column=resolved.column,
            state_ids=tuple(resolved.state_ids),
            source_values=MappingProxyType(deepcopy(dict(resolved.source_values))),
            case_sensitive=resolved.case_sensitive,
        )
        for ref in decl.experiment_semantics.annotations.ordered_state_spaces.by_id
        for resolved in (decl.experiment_semantics.annotations.resolve_ordered_state_space(ref=ref),)
    }
    return NotebookContext(
        experiment=experiment,
        experiment_root=decl.experiment.root,
        outputs_dir=layout.outputs_dir,
        notebooks_dir=notebooks_dir,
        pipeline_step_ids=tuple(step.id for step in workbench.pipeline),
        pipeline_steps=pipeline_steps,
        protocol_inputs=MappingProxyType(deepcopy(dict(decl.experiment_semantics.protocol.inputs or {}))),
        ordered_state_spaces=MappingProxyType(ordered_state_spaces),
    )


def resolve_effective_step_config(experiment: Experiment, step_id: str) -> CompiledStepConfig:
    """Resolve one compiled pipeline step and its effective protocol-bound configuration."""

    steps = tuple(step for step in resolve_workbench(experiment._declaration).pipeline if step.id == step_id)
    if len(steps) != 1:
        raise ConfigError(f"Expected exactly one compiled pipeline step {step_id!r}, found {len(steps)}")
    step = steps[0]
    descriptor = experiment._runtime.plugins.resolve_descriptor(step.plugin)
    projected_step = CompiledStep(
        step_id=step.id,
        plugin_id=step.plugin,
        domain=descriptor.domain,
        family=descriptor.family,
        tags=tuple(descriptor.tags),
    )
    bound_protocol = experiment._runtime.bind_protocol(experiment._declaration.experiment_semantics.protocol)
    values = bound_protocol.effective_plugin_config(
        plugin_id=step.plugin,
        step_with=dict(step.with_ or {}),
    )
    return CompiledStepConfig(
        step=projected_step,
        values=MappingProxyType(deepcopy(values)),
    )


__all__ = [
    "CompiledStep",
    "CompiledStepConfig",
    "NotebookContext",
    "NotebookDeliverables",
    "NotebookOverview",
    "OrderedStateSpaceContext",
    "build_dataframe_record_catalog",
    "build_design_treatment_summary_rows",
    "build_notebook_deliverable_selector",
    "build_notebook_overview",
    "collect_notebook_deliverables",
    "load_notebook_context",
    "render_notebook_deliverable_viewport",
    "render_notebook_overview_panel",
    "resolve_effective_step_config",
    "select_default_dataframe_record",
]
