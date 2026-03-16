from __future__ import annotations

from pathlib import Path
from typing import Any

from reader.core.errors import ConfigError
from reader.workbench.assets import resolve_notebook_template_asset, resolve_recipe_steps
from reader.workbench.decl import (
    FileInputDecl,
    PluginStepDecl,
    RecordInputDecl,
    RecordOutputDecl,
    ResourceInputDecl,
    WorkbenchDecl,
)
from reader.workbench.experiment import ResourceCatalog
from reader.workbench.graph.nodes import (
    NotebookTemplateCall,
    PluginStep,
    RecipeSource,
    Workbench,
    ensure_unique_workbench_ids,
)
from reader.workbench.graph.refs import FileRef, InputRef, OutputRef, RecordRef, ResourceRef
from reader.workbench.ontology import (
    WorkbenchPluginStepKind,
    get_workbench_surface_semantics,
)


def resolve_workbench(decl: WorkbenchDecl) -> Workbench:
    pipeline = tuple(
        _resolve_plugin_steps(
            decl,
            kind="pipeline",
            recipes=decl.pipeline.recipes,
            defaults={"reads": {}, "with": {}},
            overrides=decl.pipeline.overrides,
            specs=decl.pipeline.steps,
        )
    )
    plots = tuple(
        _resolve_plugin_steps(
            decl,
            kind="plot",
            recipes=decl.plots.recipes,
            defaults={"reads": decl.plots.defaults.reads, "with": decl.plots.defaults.with_},
            overrides=decl.plots.overrides,
            specs=decl.plots.specs,
        )
    )
    exports = tuple(
        _resolve_plugin_steps(
            decl,
            kind="export",
            recipes=decl.exports.recipes,
            defaults={"reads": decl.exports.defaults.reads, "with": decl.exports.defaults.with_},
            overrides=decl.exports.overrides,
            specs=decl.exports.specs,
        )
    )
    notebooks = tuple(_resolve_notebook_specs(decl))
    ensure_unique_workbench_ids(pipeline, plots, exports, notebooks)
    return Workbench(pipeline=pipeline, plots=plots, exports=exports, notebooks=notebooks)


def materialize_workbench(decl: WorkbenchDecl) -> dict[str, list[dict[str, Any]]]:
    workbench = resolve_workbench(decl)
    return {
        "pipeline": [item.to_dict() for item in workbench.pipeline],
        "plots": [item.to_dict() for item in workbench.plots],
        "exports": [item.to_dict() for item in workbench.exports],
        "notebooks": [item.to_dict() for item in workbench.notebooks],
    }


def select_workbench_specs(
    specs: list[PluginStep] | list[NotebookTemplateCall],
    *,
    only: list[str],
    exclude: list[str],
    kind_label: str,
) -> list[PluginStep] | list[NotebookTemplateCall]:
    ids = [item.id for item in specs]
    available = set(ids)
    if only:
        only_ids = set(only)
        missing = sorted(only_ids - available)
        if missing:
            raise ConfigError(f"Unknown {kind_label} id(s): {missing}.")
        selected = [item for item in specs if item.id in only_ids]
    else:
        selected = list(specs)
    if exclude:
        exclude_ids = set(exclude)
        missing = sorted(exclude_ids - available)
        if missing:
            raise ConfigError(f"Unknown {kind_label} id(s): {missing}.")
        selected = [item for item in selected if item.id not in exclude_ids]
    return selected


def normalize_input_binding(
    binding: Any,
    *,
    root: Path,
    resources: ResourceCatalog,
    section: str,
    step_id: str,
    key: str,
) -> InputRef:
    if isinstance(binding, RecordInputDecl):
        return RecordRef(record_id=binding.record_id)
    if isinstance(binding, FileInputDecl):
        path = Path(binding.path).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        return FileRef(path=path)
    if isinstance(binding, ResourceInputDecl):
        try:
            resource = resources.require_file(binding.resource_id)
        except ValueError as err:
            raise ConfigError(
                f"{section} {step_id}: reads '{key}' references unknown resource '{binding.resource_id}'."
            ) from err
        return ResourceRef(resource_id=binding.resource_id, path=resource.path)
    raise ConfigError(f"{section} {step_id}: reads '{key}' uses unsupported binding type {type(binding).__name__}")


def normalize_output_binding(binding: RecordOutputDecl) -> OutputRef:
    return OutputRef(record_id=binding.record_id)


def _resolve_plugin_steps(
    decl: WorkbenchDecl,
    *,
    kind: WorkbenchPluginStepKind,
    recipes: list[Any],
    defaults: dict[str, Any],
    overrides: dict[str, Any],
    specs: list[PluginStepDecl] | tuple[PluginStepDecl, ...],
) -> list[PluginStep]:
    semantics = get_workbench_surface_semantics(kind)
    root = decl.experiment.root
    resources = decl.experiment_semantics.resources
    raw_steps: list[PluginStepDecl] = []

    for recipe in recipes or []:
        recipe_name, recipe_with = _normalize_recipe_call(recipe)
        expanded = resolve_recipe_steps(recipe_name, with_args=recipe_with)
        raw_steps.extend(
            PluginStepDecl(
                id=entry.id,
                plugin=entry.plugin,
                reads=dict(entry.reads or {}),
                writes=dict(entry.writes or {}),
                with_=dict(entry.with_ or {}),
                source_recipe=entry.source_recipe,
            )
            for entry in expanded
        )

    for entry in specs or []:
        raw_steps.append(entry)

    if not raw_steps:
        return []

    default_reads = defaults.get("reads") or {}
    default_with = defaults.get("with") or {}
    if not isinstance(default_reads, dict):
        raise ConfigError(f"{semantics.section}.defaults.reads must be a mapping")
    if not isinstance(default_with, dict):
        raise ConfigError(f"{semantics.section}.defaults.with must be a mapping")

    finalized: list[PluginStepDecl] = []
    for step in raw_steps:
        step_id = step.id
        if not step_id or not isinstance(step_id, str):
            raise ConfigError(f"Every {semantics.section} spec must include an id.")
        plugin = step.plugin
        if not plugin or not isinstance(plugin, str):
            raise ConfigError(f"{semantics.section} {step_id}: plugin must be a non-empty string")
        if "/" not in plugin:
            raise ConfigError(f"{semantics.section} {step_id}: plugin must be 'category/key'")
        category = plugin.split("/", 1)[0]
        expected_category = semantics.expected_plugin_category
        if kind == "pipeline" and category in {"plot", "export"}:
            raise ConfigError(f"pipeline {step_id}: plot/export plugins are not allowed in pipeline.")
        if expected_category is not None and category != expected_category:
            raise ConfigError(f"{semantics.section} {step_id}: plugin must be {expected_category}/*")

        finalized.append(
            PluginStepDecl(
                id=step_id,
                plugin=plugin,
                reads={**default_reads, **dict(step.reads or {})},
                with_={**default_with, **dict(step.with_ or {})},
                writes=dict(step.writes or {}),
                source_recipe=step.source_recipe,
            )
        )

    if overrides:
        if not isinstance(overrides, dict):
            raise ConfigError(f"{semantics.section}.overrides must be a mapping of id -> overrides")
        ids = {item.id for item in finalized}
        unknown = sorted(set(overrides) - ids)
        if unknown:
            raise ConfigError(
                f"{semantics.section}.overrides reference unknown id(s): {unknown}. "
                "Check recipe-expanded ids or remove stale overrides."
            )
        for index, step in enumerate(finalized):
            step_id = step.id
            if step_id not in overrides:
                continue
            finalized[index] = _apply_step_override(step, override=overrides[step_id], section=semantics.section)

    seen: set[str] = set()
    resolved: list[PluginStep] = []
    for step in finalized:
        step_id = step.id
        if step_id in seen:
            raise ConfigError(f"{semantics.section} contains duplicate spec id(s): {step_id}")
        seen.add(step_id)
        resolved.append(
            PluginStep(
                kind=kind,
                id=step_id,
                plugin=step.plugin,
                reads={
                    key: normalize_input_binding(
                        value,
                        root=root,
                        resources=resources,
                        section=semantics.section,
                        step_id=step_id,
                        key=key,
                    )
                    for key, value in (step.reads or {}).items()
                },
                with_=dict(step.with_ or {}),
                writes={key: normalize_output_binding(value) for key, value in (step.writes or {}).items()},
                source_recipe=_normalize_recipe_source(step.source_recipe),
            )
        )
    return resolved


def _resolve_notebook_specs(decl: WorkbenchDecl) -> list[NotebookTemplateCall]:
    semantics = get_workbench_surface_semantics("notebook")
    raw_specs = list(decl.notebooks.specs or [])
    if not raw_specs:
        return []

    seen: set[str] = set()
    resolved: list[NotebookTemplateCall] = []
    for entry in raw_specs:
        step_id = entry.id
        if not step_id or not isinstance(step_id, str):
            raise ConfigError(f"Every {semantics.section} spec must include an id.")
        template = entry.template
        if not template or not isinstance(template, str):
            raise ConfigError(f"{semantics.section} {step_id}: template must be a non-empty string")
        descriptor = resolve_notebook_template_asset(template)
        if step_id in seen:
            raise ConfigError(f"{semantics.section} contains duplicate spec id(s): {step_id}")
        seen.add(step_id)
        resolved.append(NotebookTemplateCall(id=step_id, template=descriptor.template))
    return resolved


def _normalize_recipe_call(raw: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(raw, str):
        return raw, {}
    if hasattr(raw, "recipe") and hasattr(raw, "with_"):
        return str(raw.recipe), dict(raw.with_ or {})
    if isinstance(raw, dict):
        recipe = raw.get("recipe")
        with_block = raw.get("with", {}) or {}
        if not isinstance(recipe, str) or not recipe.strip():
            raise ConfigError("Recipe call recipe must be a non-empty string.")
        if not isinstance(with_block, dict):
            raise ConfigError(f"Recipe call for {recipe!r}: with must be a mapping.")
        return recipe, dict(with_block)
    raise ConfigError(f"Unsupported recipe call type: {type(raw).__name__}")


def _apply_step_override(step: PluginStepDecl, *, override: dict[str, Any], section: str) -> PluginStepDecl:
    step_id = override.get("id", step.id)
    if step_id != step.id:
        raise ConfigError(f"{section}.overrides for '{step.id}' cannot change the id.")
    plugin = override.get("plugin", step.plugin)
    if not isinstance(plugin, str) or not plugin.strip():
        raise ConfigError(f"{section}.overrides.{step.id}.plugin must be a non-empty string when provided")
    reads = dict(step.reads or {})
    writes = dict(step.writes or {})
    with_block = dict(step.with_ or {})
    if "reads" in override:
        reads_override = override["reads"]
        if not isinstance(reads_override, dict):
            raise ConfigError(f"{section}.overrides.{step.id}.reads must be a mapping")
        reads = {**reads, **reads_override}
    if "writes" in override:
        writes_override = override["writes"]
        if not isinstance(writes_override, dict):
            raise ConfigError(f"{section}.overrides.{step.id}.writes must be a mapping")
        writes = {**writes, **writes_override}
    if "with" in override:
        with_override = override["with"]
        if not isinstance(with_override, dict):
            raise ConfigError(f"{section}.overrides.{step.id}.with must be a mapping")
        with_block = {**with_block, **with_override}
    return PluginStepDecl(
        id=step.id,
        plugin=plugin,
        reads=reads,
        writes=writes,
        with_=with_block,
        source_recipe=step.source_recipe,
    )


def _normalize_recipe_source(source: Any) -> RecipeSource | None:
    if source is None:
        return None
    recipe = getattr(source, "recipe", None)
    with_block = getattr(source, "with_", None)
    if not isinstance(recipe, str):
        raise ConfigError(f"Unsupported recipe source type: {type(source).__name__}")
    return RecipeSource(recipe=recipe, with_=dict(with_block or {}))
