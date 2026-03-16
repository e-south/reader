from __future__ import annotations

import pytest

from reader.core.errors import ConfigError
from reader.workbench.ontology import WorkbenchRecipeSemantics
from reader.workbench.recipes.registry import _build_recipe_catalog


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
