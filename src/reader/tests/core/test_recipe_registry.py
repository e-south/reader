from __future__ import annotations

import pytest

from reader.errors import ConfigError
from reader.workbench.ontology import WorkbenchRecipeSemantics
from reader.workbench.recipes.registry import RECIPES, _build_recipe_catalog, resolve_recipe_steps


def test_build_recipe_catalog_rejects_duplicate_names() -> None:
    semantics = WorkbenchRecipeSemantics(
        kind="recipe",
        domain="generic",
        family="test_family",
        summary="one",
    )
    with pytest.raises(ConfigError, match="Duplicate recipe 'shared/recipe'"):
        _build_recipe_catalog(
            ("first", {"shared/recipe": {"semantics": semantics, "steps": []}}),
            ("second", {"shared/recipe": {"semantics": semantics, "steps": []}}),
        )


def test_registered_recipes_resolve_with_typed_source_recipe() -> None:
    for recipe_name, info in RECIPES.items():
        resolved = resolve_recipe_steps(recipe_name, with_args={"audit": True})

        assert resolved, f"{recipe_name}: recipe should resolve to at least one step"
        assert len(resolved) == len(info["steps"])
        for step in resolved:
            assert step.source_recipe is not None, f"{recipe_name}: missing typed recipe provenance"
            assert step.source_recipe.recipe == recipe_name
            assert step.source_recipe.with_ == {"audit": True}
