import importlib

__all__ = [
    "RECIPES",
    "RecipeCatalog",
    "RecipeDescriptor",
    "describe_recipe",
    "list_recipes",
    "resolve_recipe",
    "resolve_recipe_asset",
]


def __getattr__(name: str):
    if name in {
        "RECIPES",
        "RecipeCatalog",
        "RecipeDescriptor",
        "describe_recipe",
        "list_recipes",
        "resolve_recipe",
        "resolve_recipe_asset",
    }:
        _registry = importlib.import_module("reader.workbench.recipes.registry")
        return getattr(_registry, name)
    raise AttributeError(name)
