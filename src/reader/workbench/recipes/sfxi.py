"""
--------------------------------------------------------------------------------
<reader project>
src/reader/workbench/recipes/sfxi.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from reader.workbench.ontology import WorkbenchRecipeSemantics
from reader.workbench.recipes._helpers import recipe_step

SFXI_RECIPES: dict[str, dict[str, Any]] = {
    "sfxi/promote": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="logic",
            family="sfxi_preparation",
            summary="Promote tidy table to annotated plate-reader data for SFXI.",
            tags=("pipeline", "sfxi"),
        ),
        "steps": [
            recipe_step(
                id="promote_to_tidy_plus_map",
                plugin="validator/to_tidy_plus_map",
                reads={"df": {"record": "ratio_yfp_od600/df"}},
                with_={"synthesize_batch": True},
            )
        ],
    },
    "sfxi/vec8": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="logic",
            family="sfxi_analysis",
            summary="Compute SFXI vec8 labels from annotated plate-reader data.",
            tags=("pipeline", "sfxi"),
        ),
        "steps": [
            recipe_step(
                id="sfxi_vec8",
                plugin="transform/sfxi",
                reads={"df": {"record": "promote_to_tidy_plus_map/df"}},
                with_={
                    "response": {"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
                    "design_by": ["design_id"],
                    "logic_map_ref": "induction_logic",
                },
            )
        ],
    },
}
