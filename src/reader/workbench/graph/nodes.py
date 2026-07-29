from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from reader.errors import ConfigError
from reader.workbench.graph.refs import (
    InputRef,
    OutputRef,
    input_ref_to_dict,
    output_ref_to_dict,
)
from reader.workbench.ontology import (
    WorkbenchItemKind,
    WorkbenchPluginStepKind,
    WorkbenchSurfaceSemantics,
    get_workbench_surface_semantics,
)


@dataclass(frozen=True)
class RecipeSource:
    recipe: str
    with_: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginStep:
    kind: WorkbenchPluginStepKind
    id: str
    plugin: str
    reads: dict[str, InputRef] = field(default_factory=dict)
    with_: dict[str, Any] = field(default_factory=dict)
    writes: dict[str, OutputRef] = field(default_factory=dict)
    source_recipe: RecipeSource | None = None

    @property
    def semantics(self) -> WorkbenchSurfaceSemantics:
        return get_workbench_surface_semantics(self.kind)

    @property
    def plugin_category(self) -> str:
        return self.plugin.split("/", 1)[0] if "/" in self.plugin else self.plugin

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "plugin": self.plugin,
            "reads": {key: input_ref_to_dict(value) for key, value in (self.reads or {}).items()},
            "with": dict(self.with_ or {}),
            "writes": {key: output_ref_to_dict(value) for key, value in (self.writes or {}).items()},
        }
        if self.source_recipe is not None:
            payload["source_recipe"] = {
                "recipe": self.source_recipe.recipe,
                "with": dict(self.source_recipe.with_ or {}),
            }
        return payload


@dataclass(frozen=True)
class NotebookTemplateCall:
    id: str
    template: str

    @property
    def semantics(self) -> WorkbenchSurfaceSemantics:
        return get_workbench_surface_semantics("notebook")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "template": self.template}


WorkbenchItem = PluginStep | NotebookTemplateCall


@dataclass(frozen=True)
class Workbench:
    pipeline: tuple[PluginStep, ...] = ()
    plots: tuple[PluginStep, ...] = ()
    exports: tuple[PluginStep, ...] = ()
    notebooks: tuple[NotebookTemplateCall, ...] = ()

    def by_kind(self, kind: WorkbenchItemKind) -> tuple[WorkbenchItem, ...]:
        if kind == "pipeline":
            return self.pipeline
        if kind == "plot":
            return self.plots
        if kind == "export":
            return self.exports
        return self.notebooks

    def all_specs(self) -> tuple[WorkbenchItem, ...]:
        return self.pipeline + self.plots + self.exports + self.notebooks

    def plugin_steps(self) -> tuple[PluginStep, ...]:
        return self.pipeline + self.plots + self.exports

    def plugin_categories(self) -> set[str]:
        return {item.plugin_category for item in self.plugin_steps()}

    def counts(self) -> dict[WorkbenchItemKind, int]:
        return {
            "pipeline": len(self.pipeline),
            "plot": len(self.plots),
            "export": len(self.exports),
            "notebook": len(self.notebooks),
        }


def ensure_unique_workbench_ids(*collections: Sequence[WorkbenchItem]) -> None:
    ids: list[str] = [item.id for collection in collections for item in collection if item.id]
    dupes = sorted(item_id for item_id, count in Counter(ids).items() if count > 1)
    if dupes:
        raise ConfigError(f"Duplicate step/spec id(s) across pipeline/plots/exports/notebooks: {dupes}")
