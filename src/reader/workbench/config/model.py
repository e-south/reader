from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from reader.errors import ConfigError


class ExperimentSpec(BaseModel):
    id: str
    title: str | None = None

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
    kind: str
    path: str

    model_config = {"extra": "forbid"}


class ResourcesSpec(BaseModel):
    by_id: dict[str, ResourceSpec] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class AnnotationLabelSpec(BaseModel):
    source: str
    values: dict[str, str] = Field(default_factory=dict)
    output: str | None = None

    model_config = {"extra": "forbid"}


class AnnotationOrderSpec(BaseModel):
    column: str
    values: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class AnnotationCollectionSpec(BaseModel):
    column: str
    items: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class AnnotationLogicMapSpec(BaseModel):
    column: str
    corners: dict[str, str]
    case_sensitive: bool = True

    model_config = {"extra": "forbid"}


class AnnotationSpec(BaseModel):
    labels: dict[str, AnnotationLabelSpec] = Field(default_factory=dict)
    orders: dict[str, AnnotationOrderSpec] = Field(default_factory=dict)
    collections: dict[str, AnnotationCollectionSpec] = Field(default_factory=dict)
    logic_maps: dict[str, AnnotationLogicMapSpec] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class NotebookDeliverablesSpec(BaseModel):
    template: str | None = None

    model_config = {"extra": "forbid"}


class SurfaceDeliverablesSpec(BaseModel):
    profile: str | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    settings: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class DeliverablesSpec(BaseModel):
    notebook: NotebookDeliverablesSpec = Field(default_factory=NotebookDeliverablesSpec)
    plots: SurfaceDeliverablesSpec = Field(default_factory=SurfaceDeliverablesSpec)
    exports: SurfaceDeliverablesSpec = Field(default_factory=SurfaceDeliverablesSpec)

    model_config = {"extra": "forbid"}


class ProtocolBindingSpec(BaseModel):
    id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] = Field(default_factory=dict)
    deliverables: DeliverablesSpec = Field(default_factory=DeliverablesSpec)

    model_config = {"extra": "forbid"}

    @field_validator("id", mode="after")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ConfigError("protocol.id must be a non-empty string")
        return v


class ReaderSpec(BaseModel):
    schema_: str = Field(alias="schema")
    experiment: ExperimentSpec
    protocol: ProtocolBindingSpec
    paths: PathsSpec = Field(default_factory=PathsSpec)
    plotting: PlottingSpec = Field(default_factory=PlottingSpec)
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)
    annotations: AnnotationSpec = Field(default_factory=AnnotationSpec)

    model_config = {"extra": "forbid"}

    @field_validator("schema_", mode="after")
    @classmethod
    def _validate_schema(cls, v: str) -> str:
        if v != "reader/v6":
            raise ConfigError("Config schema must be 'reader/v6'. This repo only supports reader/v6.")
        return v

    @classmethod
    def load(cls, path: Path) -> ReaderSpec:
        load_module = import_module("reader.workbench.config.load")
        return load_module.load_reader_spec(path, cls=cls)
