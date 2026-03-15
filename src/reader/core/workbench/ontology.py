from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PluginCategory = Literal["ingest", "merge", "transform", "plot", "export", "validator"]
WorkbenchSpecKind = Literal["pipeline", "plot", "export", "notebook"]
WorkbenchProducerKind = Literal["pipeline", "plot", "export", "notebook"]
WorkbenchRecordKind = Literal["dataframe_artifact", "file_bundle"]


@dataclass(frozen=True)
class PluginSemantics:
    """Small ontology for the workbench plugin surface."""

    category: PluginCategory
    domain: str
    family: str
    summary: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkbenchSpecSemantics:
    """Semantic descriptor for a workbench spec surface."""

    kind: WorkbenchSpecKind
    section: str
    title: str
    item_label: str
    items_label: str
    uses_category: str | None
    plugin_backed: bool = True


_WORKBENCH_SPEC_SEMANTICS: dict[WorkbenchSpecKind, WorkbenchSpecSemantics] = {
    "pipeline": WorkbenchSpecSemantics(
        kind="pipeline",
        section="pipeline",
        title="Pipeline",
        item_label="pipeline step",
        items_label="pipeline steps",
        uses_category=None,
    ),
    "plot": WorkbenchSpecSemantics(
        kind="plot",
        section="plots",
        title="Plots",
        item_label="plot spec",
        items_label="plot specs",
        uses_category="plot",
    ),
    "export": WorkbenchSpecSemantics(
        kind="export",
        section="exports",
        title="Exports",
        item_label="export spec",
        items_label="export specs",
        uses_category="export",
    ),
    "notebook": WorkbenchSpecSemantics(
        kind="notebook",
        section="notebooks",
        title="Notebooks",
        item_label="notebook spec",
        items_label="notebook specs",
        uses_category="notebook",
        plugin_backed=False,
    ),
}


def get_workbench_spec_semantics(kind: WorkbenchSpecKind) -> WorkbenchSpecSemantics:
    return _WORKBENCH_SPEC_SEMANTICS[kind]
