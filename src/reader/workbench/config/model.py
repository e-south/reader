from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from reader.errors import ConfigError


class ExperimentSpec(BaseModel):
    id: str
    title: str | None = None
    lifecycle: str = "active"

    model_config = {"extra": "forbid"}

    @field_validator("id", mode="after")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ConfigError("experiment.id must be a non-empty string")
        return v

    @field_validator("lifecycle", mode="after")
    @classmethod
    def _validate_lifecycle(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ConfigError("experiment.lifecycle must be a non-empty string")
        lifecycle = v.strip().lower()
        allowed = {"active", "draft", "template"}
        if lifecycle not in allowed:
            raise ConfigError(f"experiment.lifecycle must be one of: {', '.join(sorted(allowed))}")
        return lifecycle


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


class AnnotationOrderedStateSpaceSpec(BaseModel):
    column: str
    state_order: list[str]
    values: dict[str, str]
    case_sensitive: bool = True

    model_config = {"extra": "forbid"}

    @field_validator("column", mode="after")
    @classmethod
    def _validate_column(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ordered state-space column must be a non-empty string")
        return value.strip()

    @model_validator(mode="after")
    def _validate_states(self) -> AnnotationOrderedStateSpaceSpec:
        if not self.state_order:
            raise ValueError("state_order must be a non-empty list")
        state_ids = self.state_order
        if any(not state_id.strip() for state_id in state_ids):
            raise ValueError("state_order must contain non-empty strings")
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("state ids must be unique")
        if set(self.values) != set(state_ids):
            raise ValueError("values must have exactly the ids declared by state_order")
        source_values = [self.values[state_id] for state_id in state_ids]
        if any(not value.strip() for value in source_values):
            raise ValueError("values must map state ids to non-empty strings")
        compared = source_values if self.case_sensitive else [value.strip().casefold() for value in source_values]
        if len(set(compared)) != len(compared):
            sensitivity = "true" if self.case_sensitive else "false"
            raise ValueError(f"source values must be unique under case_sensitive={sensitivity}")
        return self


class AnnotationSpec(BaseModel):
    labels: dict[str, AnnotationLabelSpec] = Field(default_factory=dict)
    orders: dict[str, AnnotationOrderSpec] = Field(default_factory=dict)
    collections: dict[str, AnnotationCollectionSpec] = Field(default_factory=dict)
    ordered_state_spaces: dict[str, AnnotationOrderedStateSpaceSpec] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class NotebookOutputsSpec(BaseModel):
    template: str | None = None

    model_config = {"extra": "forbid"}


class PlotOutputsSpec(BaseModel):
    profile: str | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    views: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class ExportOutputsSpec(BaseModel):
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class OutputsSpec(BaseModel):
    notebook: NotebookOutputsSpec = Field(default_factory=NotebookOutputsSpec)
    plots: PlotOutputsSpec = Field(default_factory=PlotOutputsSpec)
    exports: ExportOutputsSpec = Field(default_factory=ExportOutputsSpec)

    model_config = {"extra": "forbid"}


class ProtocolBindingSpec(BaseModel):
    id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] = Field(default_factory=dict)
    outputs: OutputsSpec = Field(default_factory=OutputsSpec)

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
        if v != "reader/v8":
            raise ConfigError("Config schema must be 'reader/v8'. This repo only supports reader/v8.")
        return v

    @classmethod
    def load(cls, path: Path) -> ReaderSpec:
        load_module = import_module("reader.workbench.config.load")
        return load_module.load_reader_spec(path, cls=cls)
