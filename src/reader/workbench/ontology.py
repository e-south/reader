from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PluginCategory = Literal["ingest", "transform", "plot", "export", "validator"]
PluginDomain = Literal["generic", "plate_reader", "cytometry", "logic"]
WorkbenchItemKind = Literal["pipeline", "plot", "export"]
WorkbenchPluginStepKind = Literal["pipeline", "plot", "export"]
WorkbenchProducerKind = Literal["pipeline", "plot", "export"]
WorkbenchRecordKind = Literal["dataframe_artifact", "file_bundle"]
WorkbenchRefKind = Literal["plugin"]

KNOWN_PLUGIN_DOMAINS: tuple[PluginDomain, ...] = ("generic", "plate_reader", "cytometry", "logic")


def validate_plugin_domain(domain: str) -> PluginDomain:
    normalized = str(domain).strip()
    if normalized in KNOWN_PLUGIN_DOMAINS:
        return normalized  # type: ignore[return-value]
    options = ", ".join(KNOWN_PLUGIN_DOMAINS)
    raise ValueError(f"Unknown plugin domain '{domain}'. Expected one of: {options}")


@dataclass(frozen=True)
class PluginSemantics:
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
}


def get_workbench_surface_semantics(kind: WorkbenchItemKind) -> WorkbenchSurfaceSemantics:
    return _WORKBENCH_SURFACE_SEMANTICS[kind]
