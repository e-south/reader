from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reader_workbench.workbench.experiment import ExperimentSemantics


@dataclass(frozen=True)
class ExperimentDecl:
    id: str
    title: str
    lifecycle: str
    root: Path


@dataclass(frozen=True)
class RecordInputDecl:
    record_id: str


@dataclass(frozen=True)
class FileInputDecl:
    path: str


@dataclass(frozen=True)
class ResourceInputDecl:
    resource_id: str


@dataclass(frozen=True)
class RecordCollectionInputDecl:
    resource_ids: tuple[str, ...]


InputBindingDecl = RecordInputDecl | FileInputDecl | ResourceInputDecl | RecordCollectionInputDecl


@dataclass(frozen=True)
class RecordOutputDecl:
    record_id: str


@dataclass(frozen=True)
class RecipeSourceDecl:
    recipe: str
    with_: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginStepDecl:
    id: str
    plugin: str
    reads: dict[str, InputBindingDecl] = field(default_factory=dict)
    writes: dict[str, RecordOutputDecl] = field(default_factory=dict)
    with_: dict[str, Any] = field(default_factory=dict)
    source_recipe: RecipeSourceDecl | None = None


@dataclass(frozen=True)
class PipelineDecl:
    runtime: dict[str, Any] = field(default_factory=dict)
    steps: tuple[PluginStepDecl, ...] = ()


@dataclass(frozen=True)
class SurfaceDecl:
    specs: tuple[PluginStepDecl, ...] = ()


@dataclass(frozen=True)
class WorkbenchDecl:
    experiment: ExperimentDecl
    experiment_semantics: ExperimentSemantics
    plotting_palette: str | None
    pipeline: PipelineDecl
    plots: SurfaceDecl
    exports: SurfaceDecl
    config_digest: str = ""
