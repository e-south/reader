from __future__ import annotations

from typing import Any

from reader.errors import ConfigError
from reader.workbench.decl import PluginStepDecl, RecipeSourceDecl
from reader.workbench.ontology import WorkbenchRecipeSemantics
from reader.workbench.recipes.plate_reader import PLATE_READER_RECIPES

RecipeInfo = dict[str, Any]
RecipeCatalog = dict[str, RecipeInfo]


def _build_recipe_catalog(*sources: tuple[str, dict[str, dict[str, Any]]]) -> RecipeCatalog:
    catalog: RecipeCatalog = {}
    owners: dict[str, str] = {}
    for owner, source in sources:
        for recipe, info in source.items():
            previous = owners.get(recipe)
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
            catalog[recipe] = {"semantics": semantics, "steps": tuple(steps)}
            owners[recipe] = owner
    return catalog


RECIPES = _build_recipe_catalog(
    ("plate_reader", PLATE_READER_RECIPES),
)


def resolve_recipe_steps(name: str, *, with_args: dict[str, Any] | None = None) -> list[PluginStepDecl]:
    try:
        info = RECIPES[name]
    except KeyError:
        options = ", ".join(sorted(RECIPES)) or "—"
        raise ConfigError(f"Unknown internal recipe {name!r}. Available recipes: {options}") from None
    source = RecipeSourceDecl(recipe=name, with_=dict(with_args or {}))
    return [_attach_recipe_source(step, source=source) for step in info["steps"]]


def _attach_recipe_source(step: PluginStepDecl, *, source: RecipeSourceDecl) -> PluginStepDecl:
    return PluginStepDecl(
        id=step.id,
        plugin=step.plugin,
        reads=dict(step.reads or {}),
        writes=dict(step.writes or {}),
        with_=dict(step.with_ or {}),
        source_recipe=RecipeSourceDecl(recipe=source.recipe, with_=dict(source.with_ or {})),
    )
