from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reader.domains.semantics import PluginDomain, validate_plugin_domain

PluginCategory = Literal["ingest", "transform", "plot", "export", "validator"]
WorkbenchItemKind = Literal["pipeline", "plot", "export", "notebook"]
WorkbenchPluginStepKind = Literal["pipeline", "plot", "export"]
WorkbenchProducerKind = Literal["pipeline", "plot", "export", "notebook"]
WorkbenchRecordKind = Literal["dataframe_artifact", "file_bundle"]
WorkbenchTemplateKind = Literal["notebook"]
WorkbenchRecipeKind = Literal["recipe"]
WorkbenchRefKind = Literal["plugin", "template"]


@dataclass(frozen=True)
class PluginSemantics:
    domain: PluginDomain
    family: str
    summary: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", validate_plugin_domain(self.domain))


@dataclass(frozen=True)
class WorkbenchTemplateSemantics:
    kind: WorkbenchTemplateKind
    domain: PluginDomain
    family: str
    summary: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", validate_plugin_domain(self.domain))


@dataclass(frozen=True)
class WorkbenchRecipeSemantics:
    kind: WorkbenchRecipeKind
    domain: PluginDomain
    family: str
    summary: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", validate_plugin_domain(self.domain))


@dataclass(frozen=True)
class WorkbenchSurfaceSemantics:
    kind: WorkbenchItemKind
    section: str
    title: str
    item_label: str
    items_label: str
    ref_kind: WorkbenchRefKind
    expected_plugin_category: str | None = None


_WORKBENCH_SURFACE_SEMANTICS: dict[WorkbenchItemKind, WorkbenchSurfaceSemantics] = {
    "pipeline": WorkbenchSurfaceSemantics(
        kind="pipeline",
        section="pipeline",
        title="Pipeline",
        item_label="pipeline step",
        items_label="pipeline steps",
        ref_kind="plugin",
    ),
    "plot": WorkbenchSurfaceSemantics(
        kind="plot",
        section="plots",
        title="Plots",
        item_label="plot spec",
        items_label="plot specs",
        ref_kind="plugin",
        expected_plugin_category="plot",
    ),
    "export": WorkbenchSurfaceSemantics(
        kind="export",
        section="exports",
        title="Exports",
        item_label="export spec",
        items_label="export specs",
        ref_kind="plugin",
        expected_plugin_category="export",
    ),
    "notebook": WorkbenchSurfaceSemantics(
        kind="notebook",
        section="notebooks",
        title="Notebooks",
        item_label="notebook spec",
        items_label="notebook specs",
        ref_kind="template",
    ),
}


def get_workbench_surface_semantics(kind: WorkbenchItemKind) -> WorkbenchSurfaceSemantics:
    return _WORKBENCH_SURFACE_SEMANTICS[kind]
