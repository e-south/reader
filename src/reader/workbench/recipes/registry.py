from __future__ import annotations

from typing import Any

from reader.core.errors import ConfigError
from reader.workbench.assets import (
    AssetCatalog,
    AssetDescriptor,
    describe_recipe_asset,
    list_recipe_assets,
    recipe_asset_catalog,
    resolve_recipe_asset,
    resolve_recipe_steps,
)
from reader.workbench.ontology import WorkbenchRecipeSemantics

RecipeCatalog = AssetCatalog
RecipeDescriptor = AssetDescriptor
RECIPES = recipe_asset_catalog()


def _build_recipe_catalog(*sources: tuple[str, dict[str, dict[str, Any]]]) -> RecipeCatalog:
    descriptors: list[RecipeDescriptor] = []
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
            descriptors.append(
                RecipeDescriptor(
                    kind="recipe",
                    name=recipe,
                    domain=semantics.domain,
                    family=semantics.family,
                    summary=semantics.summary,
                    tags=tuple(semantics.tags),
                    steps=tuple(steps),
                )
            )
            owners[recipe] = owner
    return RecipeCatalog(descriptors)


def list_recipes(family: str | None = None) -> list[tuple[str, str]]:
    return list_recipe_assets(family=family)


def resolve_recipe(name: str, *, with_args: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return resolve_recipe_steps(name, with_args=with_args)


def describe_recipe(name: str) -> dict[str, Any]:
    return describe_recipe_asset(name)


__all__ = [
    "RECIPES",
    "RecipeCatalog",
    "RecipeDescriptor",
    "_build_recipe_catalog",
    "describe_recipe",
    "list_recipes",
    "resolve_recipe",
    "resolve_recipe_asset",
]
