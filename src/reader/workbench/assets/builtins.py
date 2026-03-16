from __future__ import annotations

from typing import Any

import reader.workbench.notebooks.templates as notebook_templates
from reader.core.errors import ConfigError
from reader.workbench.decl import PluginStepDecl, RecipeSourceDecl
from reader.workbench.ontology import WorkbenchRecipeSemantics, WorkbenchTemplateSemantics
from reader.workbench.recipes.plate_reader import PLATE_READER_RECIPES
from reader.workbench.recipes.plots import PLOT_RECIPES
from reader.workbench.recipes.sfxi import SFXI_RECIPES

from .types import AssetCapabilities, AssetCatalog, AssetDescriptor, AssetRequirement


def _descriptor_from_recipe(
    *,
    recipe: str,
    semantics: WorkbenchRecipeSemantics,
    steps: list[PluginStepDecl],
) -> AssetDescriptor:
    normalized_steps = tuple(step for step in steps)
    return AssetDescriptor(
        kind="recipe",
        name=recipe,
        domain=semantics.domain,
        family=semantics.family,
        summary=semantics.summary,
        tags=tuple(semantics.tags),
        steps=normalized_steps,
    )


def _descriptor_from_template(
    *,
    template: str,
    body: str,
    semantics: WorkbenchTemplateSemantics,
    capabilities: AssetCapabilities,
) -> AssetDescriptor:
    return AssetDescriptor(
        kind="template",
        name=template,
        domain=semantics.domain,
        family=semantics.family,
        summary=semantics.summary,
        tags=tuple(semantics.tags),
        body=body,
        capabilities=capabilities,
    )


def _build_static_asset_catalog() -> AssetCatalog:
    descriptors: list[AssetDescriptor] = []
    owners: dict[tuple[str, str], str] = {}

    recipe_sources = (
        ("plate_reader", PLATE_READER_RECIPES),
        ("sfxi", SFXI_RECIPES),
        ("plots", PLOT_RECIPES),
    )
    for owner, source in recipe_sources:
        for recipe, info in source.items():
            key = ("recipe", recipe)
            previous = owners.get(key)
            if previous is not None:
                raise ConfigError(f"Duplicate recipe {recipe!r} declared in both {previous} and {owner}.")
            semantics = info.get("semantics")
            if not isinstance(semantics, WorkbenchRecipeSemantics):
                raise ConfigError(f"Recipe {recipe!r} in {owner} must declare WorkbenchRecipeSemantics.")
            steps = info.get("steps", [])
            if not isinstance(steps, list):
                raise ConfigError(f"Recipe {recipe!r} in {owner} must declare a list of steps.")
            if not all(isinstance(step, PluginStepDecl) for step in steps):
                raise ConfigError(f"Recipe {recipe!r} in {owner} must declare PluginStepDecl steps.")
            descriptors.append(_descriptor_from_recipe(recipe=recipe, semantics=semantics, steps=steps))
            owners[key] = owner

    template_descriptors = [
        _descriptor_from_template(
            template="notebook/eda",
            body=notebook_templates.EXPERIMENT_NOTEBOOK_EDA_TEMPLATE,
            semantics=WorkbenchTemplateSemantics(
                kind="notebook",
                domain="generic",
                family="record_explorer",
                summary="Minimal dataframe-record explorer.",
                tags=("eda", "records", "microplate"),
            ),
            capabilities=AssetCapabilities(
                default_for=("has_plots",),
                supports_plot_filters=True,
                inject_plot_specs=True,
            ),
        ),
        _descriptor_from_template(
            template="notebook/basic",
            body=notebook_templates.EXPERIMENT_EDA_BASIC_TEMPLATE,
            semantics=WorkbenchTemplateSemantics(
                kind="notebook",
                domain="generic",
                family="record_explorer",
                summary="Minimal dataframe-record explorer with design/treatment table and parquet preview.",
                tags=("eda", "records"),
            ),
            capabilities=AssetCapabilities(default_for=("fallback",)),
        ),
        _descriptor_from_template(
            template="notebook/microplate",
            body=notebook_templates.EXPERIMENT_EDA_MICROPLATE_TEMPLATE,
            semantics=WorkbenchTemplateSemantics(
                kind="notebook",
                domain="plate_reader",
                family="record_explorer",
                summary="Minimal dataframe-record explorer (same scaffold as notebook/basic).",
                tags=("eda", "microplate"),
            ),
            capabilities=AssetCapabilities(),
        ),
        _descriptor_from_template(
            template="notebook/cytometry",
            body=notebook_templates.EXPERIMENT_EDA_CYTOMETRY_TEMPLATE,
            semantics=WorkbenchTemplateSemantics(
                kind="notebook",
                domain="cytometry",
                family="cytometry_eda",
                summary="Cytometry EDA scaffold (FSC/SSC scatter + fluorophore histograms).",
                tags=("eda", "cytometry"),
            ),
            capabilities=AssetCapabilities(default_for=("has_cytometry",)),
        ),
        _descriptor_from_template(
            template="notebook/sfxi_eda",
            body=notebook_templates.EXPERIMENT_SFXI_EDA_TEMPLATE,
            semantics=WorkbenchTemplateSemantics(
                kind="notebook",
                domain="logic",
                family="logic_summary",
                summary="SFXI vec8 explorer (EDA scaffold + time slice → corners → vec8).",
                tags=("eda", "sfxi", "logic"),
            ),
            capabilities=AssetCapabilities(
                requires_any=(
                    AssetRequirement(tag="sfxi"),
                    AssetRequirement(record_contract="plate_reader.annotated.v1"),
                    AssetRequirement(record_contract_prefix="sfxi.vec8."),
                )
            ),
        ),
    ]
    for descriptor in template_descriptors:
        key = ("template", descriptor.name)
        previous = owners.get(key)
        if previous is not None:
            raise ConfigError(f"Duplicate template {descriptor.name!r} declared in both {previous} and notebooks.")
        descriptors.append(descriptor)
        owners[key] = "notebooks"
    return AssetCatalog(descriptors)


_STATIC_ASSET_CATALOG = _build_static_asset_catalog()


def static_asset_catalog() -> AssetCatalog:
    return _STATIC_ASSET_CATALOG


def recipe_asset_catalog() -> AssetCatalog:
    return AssetCatalog(list(static_asset_catalog().filter(kind="recipe")))


def notebook_template_asset_catalog() -> AssetCatalog:
    return AssetCatalog(list(static_asset_catalog().filter(kind="template")))


def resolve_recipe_asset(name: str) -> AssetDescriptor:
    return static_asset_catalog().resolve(name, kind="recipe")


def resolve_notebook_template_asset(name: str) -> AssetDescriptor:
    return static_asset_catalog().resolve(name, kind="template")


def list_recipe_assets(*, family: str | None = None, domain: str | None = None) -> list[tuple[str, str]]:
    return static_asset_catalog().list(kind="recipe", family=family, domain=domain)


def list_notebook_template_assets(*, family: str | None = None, domain: str | None = None) -> list[tuple[str, str]]:
    return static_asset_catalog().list(kind="template", family=family, domain=domain)


def resolve_recipe_steps(name: str, *, with_args: dict[str, Any] | None = None) -> list[PluginStepDecl]:
    descriptor = resolve_recipe_asset(name)
    args = dict(with_args or {})
    source = RecipeSourceDecl(recipe=name, with_=args)
    if descriptor.step_builder is not None:
        steps = descriptor.step_builder(args)
        if not isinstance(steps, list):
            raise ConfigError(f"Recipe {name!r} builder must return a list of steps.")
        return [_attach_recipe_source(step, source=source) for step in steps]
    return [_attach_recipe_source(step, source=source) for step in descriptor.steps]


def describe_recipe_asset(name: str) -> dict[str, Any]:
    descriptor = resolve_recipe_asset(name)
    return {
        "recipe": descriptor.recipe,
        "domain": descriptor.domain,
        "family": descriptor.family,
        "summary": descriptor.summary,
        "tags": list(descriptor.tags),
        "steps": [_decl_step_to_dict(step) for step in descriptor.steps],
    }


def select_default_notebook_template(*, has_plots: bool, has_cytometry: bool) -> AssetDescriptor:
    contexts: list[str] = []
    if has_plots:
        contexts.append("has_plots")
    if has_cytometry:
        contexts.append("has_cytometry")
    contexts.append("fallback")
    templates = notebook_template_asset_catalog().all()
    for context in contexts:
        for descriptor in templates:
            if context in descriptor.capabilities.default_for:
                return descriptor
    raise ConfigError("No notebook template is configured for the current workbench state.")


def build_workbench_asset_catalog(*, plugin_registry: Any | None = None) -> AssetCatalog:
    if plugin_registry is None:
        raise ConfigError("build_workbench_asset_catalog() requires an explicit plugin registry.")
    plugin_assets = list(plugin_registry.catalog().all())
    return AssetCatalog(plugin_assets + list(static_asset_catalog().all()))


def _decl_step_to_dict(step: PluginStepDecl) -> dict[str, Any]:
    payload = {
        "id": step.id,
        "plugin": step.plugin,
        "reads": {key: _input_decl_to_dict(value) for key, value in (step.reads or {}).items()},
        "with": dict(step.with_ or {}),
        "writes": {key: {"record": value.record_id} for key, value in (step.writes or {}).items()},
    }
    if step.source_recipe is not None:
        payload["source_recipe"] = {"recipe": step.source_recipe.recipe, "with": dict(step.source_recipe.with_ or {})}
    return payload


def _input_decl_to_dict(value: Any) -> dict[str, str]:
    class_name = type(value).__name__
    if class_name == "RecordInputDecl":
        return {"record": value.record_id}
    if class_name == "FileInputDecl":
        return {"file": value.path}
    if class_name == "ResourceInputDecl":
        return {"resource": value.resource_id}
    raise ConfigError(f"Unsupported recipe step input declaration {class_name}")


def _attach_recipe_source(step: PluginStepDecl, *, source: RecipeSourceDecl) -> PluginStepDecl:
    return PluginStepDecl(
        id=step.id,
        plugin=step.plugin,
        reads=dict(step.reads or {}),
        writes=dict(step.writes or {}),
        with_=dict(step.with_ or {}),
        source_recipe=RecipeSourceDecl(recipe=source.recipe, with_=dict(source.with_ or {})),
    )
