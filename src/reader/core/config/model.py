from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from reader.core.errors import ConfigError


class PresetCallSpec(BaseModel):
    uses: str
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class StepSpec(BaseModel):
    id: str
    uses: str
    reads: dict[str, str] = Field(default_factory=dict)  # input label -> record label OR external handle
    writes: dict[str, str] = Field(default_factory=dict)  # output label -> record label
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ExperimentSpec(BaseModel):
    id: str
    title: str | None = None
    root: str | None = None  # internal: experiment directory

    model_config = {"extra": "forbid"}

    @field_validator("id", mode="after")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ConfigError("experiment.id must be a non-empty string")
        return v


class PathsSpec(BaseModel):
    outputs: str = "./outputs"
    plots: str = "plots"
    exports: str = "exports"
    notebooks: str = "notebooks"

    model_config = {"extra": "forbid"}


class PlottingSpec(BaseModel):
    palette: str | None = "colorblind"

    model_config = {"extra": "forbid"}


class ResourceSpec(BaseModel):
    kind: Literal["file", "directory"]
    path: str

    model_config = {"extra": "forbid"}


class ResourcesSpec(BaseModel):
    by_id: dict[str, ResourceSpec] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class AssayLabelSpec(BaseModel):
    source: str
    values: dict[str, str] = Field(default_factory=dict)
    output: str | None = None

    model_config = {"extra": "forbid"}


class AssayOrderSpec(BaseModel):
    column: str
    values: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class AssayCollectionSpec(BaseModel):
    column: str
    items: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class AssayLogicMapSpec(BaseModel):
    column: str
    corners: dict[str, str]
    case_sensitive: bool = True

    model_config = {"extra": "forbid"}


class AssaySpec(BaseModel):
    labels: dict[str, AssayLabelSpec] = Field(default_factory=dict)
    orders: dict[str, AssayOrderSpec] = Field(default_factory=dict)
    collections: dict[str, AssayCollectionSpec] = Field(default_factory=dict)
    logic_maps: dict[str, AssayLogicMapSpec] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class PipelineSpec(BaseModel):
    presets: list[str | PresetCallSpec] = Field(default_factory=list)
    runtime: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepSpec]

    model_config = {"extra": "forbid"}


class SpecDefaults(BaseModel):
    reads: dict[str, str] = Field(default_factory=dict)
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")

    model_config = {"extra": "forbid"}


class PlotSection(BaseModel):
    presets: list[str | PresetCallSpec] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    defaults: SpecDefaults = Field(default_factory=SpecDefaults)
    specs: list[StepSpec] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ExportSection(BaseModel):
    presets: list[str | PresetCallSpec] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    defaults: SpecDefaults = Field(default_factory=SpecDefaults)
    specs: list[StepSpec] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class NotebookSpec(BaseModel):
    id: str
    uses: str

    model_config = {"extra": "forbid"}


class NotebookSection(BaseModel):
    specs: list[NotebookSpec] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ReaderSpec(BaseModel):
    schema_: str = Field(alias="schema")
    experiment: ExperimentSpec
    paths: PathsSpec = Field(default_factory=PathsSpec)
    plotting: PlottingSpec = Field(default_factory=PlottingSpec)
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)
    assay: AssaySpec = Field(default_factory=AssaySpec)
    pipeline: PipelineSpec
    plots: PlotSection = Field(default_factory=PlotSection)
    exports: ExportSection = Field(default_factory=ExportSection)
    notebooks: NotebookSection = Field(default_factory=NotebookSection)

    model_config = {"extra": "forbid"}

    @field_validator("schema_", mode="after")
    @classmethod
    def _validate_schema(cls, v: str) -> str:
        if v != "reader/v3":
            raise ConfigError("Config schema must be 'reader/v3'. This repo only supports reader/v3.")
        return v

    @classmethod
    def load(cls, path: Path) -> ReaderSpec:
        load_module = import_module("reader.core.config.load")
        return load_module.load_reader_spec(path, cls=cls)
