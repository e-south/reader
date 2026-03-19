"""
--------------------------------------------------------------------------------
<reader project>
src/reader/workbench/recipes/plate_reader.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from reader.workbench.ontology import WorkbenchRecipeSemantics
from reader.workbench.recipes._helpers import recipe_step

PLATE_READER_RECIPES: dict[str, dict[str, Any]] = {
    "plate_reader/synergy_h1": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="ingest",
            summary="Synergy H1 ingest (auto-discovery).",
            tags=("pipeline", "ingest"),
        ),
        "steps": [recipe_step(id="ingest", plugin="ingest/synergy_h1")],
    },
    "plate_reader/dual_reporter_screen_base": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="workflow_base",
            summary=(
                "Dual-reporter screen base workflow: sample-map merge, design/treatment aliases, "
                "blank correction, overflow handling, and CFP/YFP/OD600 ratios."
            ),
            tags=("pipeline", "workflow", "dual_reporter"),
        ),
        "steps": [
            recipe_step(
                id="merge_map",
                plugin="transform/sample_map",
                reads={"df": {"record": "ingest/df"}, "sample_map": {"resource": "sample_map"}},
            ),
            recipe_step(id="labels", plugin="transform/assay_labels", reads={"df": {"record": "merge_map/df"}}),
            recipe_step(id="blank", plugin="transform/blank_correction", reads={"df": {"record": "labels/df"}}),
            recipe_step(
                id="overflow",
                plugin="transform/overflow_handling",
                reads={"df": {"record": "blank/df"}},
            ),
            recipe_step(
                id="ratio_yfp_cfp",
                plugin="transform/ratio",
                reads={"df": {"record": "overflow/df"}},
                with_={"name": "YFP/CFP", "numerator": "YFP", "denominator": "CFP"},
            ),
            recipe_step(
                id="ratio_cfp_od600",
                plugin="transform/ratio",
                reads={"df": {"record": "ratio_yfp_cfp/df"}},
                with_={"name": "CFP/OD600", "numerator": "CFP", "denominator": "OD600"},
            ),
            recipe_step(
                id="ratio_yfp_od600",
                plugin="transform/ratio",
                reads={"df": {"record": "ratio_cfp_od600/df"}},
                with_={"name": "YFP/OD600", "numerator": "YFP", "denominator": "OD600"},
            ),
        ],
    },
    "plate_reader/retron_sponge_screen_base": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="workflow_base",
            summary=(
                "Matched-control retron sponge base workflow: sample-map merge, assay labels, blank correction, "
                "overflow handling, and dual-reporter support ratios."
            ),
            tags=("pipeline", "workflow", "matched_control", "retron", "dual_reporter"),
        ),
        "steps": [
            recipe_step(
                id="merge_map",
                plugin="transform/sample_map",
                reads={"df": {"record": "ingest/df"}, "sample_map": {"resource": "sample_map"}},
            ),
            recipe_step(id="labels", plugin="transform/assay_labels", reads={"df": {"record": "merge_map/df"}}),
            recipe_step(id="blank", plugin="transform/blank_correction", reads={"df": {"record": "labels/df"}}),
            recipe_step(
                id="overflow",
                plugin="transform/overflow_handling",
                reads={"df": {"record": "blank/df"}},
            ),
            recipe_step(
                id="ratio_yfp_cfp",
                plugin="transform/ratio",
                reads={"df": {"record": "overflow/df"}},
                with_={"name": "YFP/CFP", "numerator": "YFP", "denominator": "CFP"},
            ),
            recipe_step(
                id="ratio_cfp_od600",
                plugin="transform/ratio",
                reads={"df": {"record": "ratio_yfp_cfp/df"}},
                with_={"name": "CFP/OD600", "numerator": "CFP", "denominator": "OD600"},
            ),
            recipe_step(
                id="ratio_yfp_od600",
                plugin="transform/ratio",
                reads={"df": {"record": "ratio_cfp_od600/df"}},
                with_={"name": "YFP/OD600", "numerator": "YFP", "denominator": "OD600"},
            ),
        ],
    },
    "plate_reader/single_reporter_screen_base": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="workflow_base",
            summary=(
                "Single-reporter screen base workflow: sample-map merge, design/treatment aliases, "
                "blank correction, overflow handling, and a configured reporter/normalizer ratio."
            ),
            tags=("pipeline", "workflow", "single_reporter"),
        ),
        "steps": [
            recipe_step(
                id="merge_map",
                plugin="transform/sample_map",
                reads={"df": {"record": "ingest/df"}, "sample_map": {"resource": "sample_map"}},
            ),
            recipe_step(id="labels", plugin="transform/assay_labels", reads={"df": {"record": "merge_map/df"}}),
            recipe_step(id="blank", plugin="transform/blank_correction", reads={"df": {"record": "labels/df"}}),
            recipe_step(
                id="overflow",
                plugin="transform/overflow_handling",
                reads={"df": {"record": "blank/df"}},
            ),
            recipe_step(
                id="ratio_reporter_normalizer",
                plugin="transform/ratio",
                reads={"df": {"record": "overflow/df"}},
                with_={"name": "Reporter/Normalizer", "numerator": "Reporter", "denominator": "Normalizer"},
            ),
        ],
    },
    "plate_reader/retron_sponge_single_reporter_base": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="workflow_base",
            summary=(
                "Matched-control retron sponge base workflow for single-reporter assays: sample-map merge, assay "
                "labels, blank correction, overflow handling, and a configured reporter/normalizer ratio."
            ),
            tags=("pipeline", "workflow", "matched_control", "retron", "single_reporter"),
        ),
        "steps": [
            recipe_step(
                id="merge_map",
                plugin="transform/sample_map",
                reads={"df": {"record": "ingest/df"}, "sample_map": {"resource": "sample_map"}},
            ),
            recipe_step(id="labels", plugin="transform/assay_labels", reads={"df": {"record": "merge_map/df"}}),
            recipe_step(id="blank", plugin="transform/blank_correction", reads={"df": {"record": "labels/df"}}),
            recipe_step(
                id="overflow",
                plugin="transform/overflow_handling",
                reads={"df": {"record": "blank/df"}},
            ),
            recipe_step(
                id="ratio_reporter_normalizer",
                plugin="transform/ratio",
                reads={"df": {"record": "overflow/df"}},
                with_={"name": "Reporter/Normalizer", "numerator": "Reporter", "denominator": "Normalizer"},
            ),
        ],
    },
    "plate_reader/sample_map": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="metadata_enrichment",
            summary="Merge plate sample map from inputs/metadata.xlsx.",
            tags=("pipeline", "metadata"),
        ),
        "steps": [
            recipe_step(
                id="merge_map",
                plugin="transform/sample_map",
                reads={"df": {"record": "ingest/df"}, "sample_map": {"resource": "sample_map"}},
            )
        ],
    },
}
