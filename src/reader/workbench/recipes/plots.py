"""
--------------------------------------------------------------------------------
<reader project>
src/reader/workbench/recipes/plots.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from reader.workbench.ontology import WorkbenchRecipeSemantics
from reader.workbench.recipes._helpers import recipe_step

PLOT_RECIPES: dict[str, dict[str, Any]] = {
    "plots/plate_reader_dual_reporter_screen_core": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="plot_set",
            summary="Core plot set for dual-reporter plate-reader screens.",
            tags=("plot", "dual_reporter"),
        ),
        "steps": [
            recipe_step(
                id="plot_time_series",
                plugin="plot/time_series",
                with_={
                    "partition": {"by": "design_id_alias"},
                    "hue": "treatment_alias",
                    "y": ["OD600", "CFP", "YFP", "YFP/CFP", "CFP/OD600", "YFP/OD600"],
                    "add_sheet_line": True,
                },
            ),
            recipe_step(
                id="snapshot_bars_by_state",
                plugin="plot/snapshot_barplot",
                with_={
                    "x": "treatment_alias",
                    "y": ["OD600", "CFP/OD600", "YFP/OD600", "YFP/CFP"],
                    "partition": {"by": "design_id_alias"},
                    "time": 14.0,
                },
            ),
            recipe_step(
                id="ts_and_snap__yfp_over_cfp",
                plugin="plot/ts_and_snap",
                with_={
                    "partition": {"by": "design_id_alias"},
                    "ts_channel": "OD600",
                    "ts_hue": "treatment_alias",
                    "ts_add_sheet_line": True,
                    "ts_mark_snap_time": True,
                    "snap_x": "treatment_alias",
                    "snap_channel": "YFP/CFP",
                    "snap_time": 14.0,
                },
            ),
        ],
    },
    "plots/plate_reader_yfp_full": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="plot_set",
            summary="YFP plate reader plot set (time series + snapshots + TS+snap).",
            tags=("plot", "yfp"),
        ),
        "steps": [
            recipe_step(
                id="plot_time_series",
                plugin="plot/time_series",
                with_={
                    "partition": {"by": "design_id"},
                    "hue": "treatment",
                    "y": ["OD600", "YFP", "YFP/CFP", "YFP/OD600"],
                    "add_sheet_line": True,
                },
            ),
            recipe_step(
                id="snapshot_bars_by_channel",
                plugin="plot/snapshot_barplot",
                with_={"x": "treatment", "y": ["OD600", "YFP/OD600"], "partition": {"by": "design_id"}, "time": 14.0},
            ),
            recipe_step(
                id="snapshot_bars_by_design_id",
                plugin="plot/snapshot_barplot",
                with_={"x": "design_id", "y": "YFP/OD600", "hue": "treatment", "time": 14.0},
            ),
            recipe_step(
                id="ts_and_snap__yfp_over_od600",
                plugin="plot/ts_and_snap",
                with_={
                    "partition": {"by": "design_id"},
                    "ts_channel": "YFP/OD600",
                    "ts_hue": "treatment",
                    "ts_add_sheet_line": True,
                    "ts_mark_snap_time": True,
                    "snap_channel": "YFP/OD600",
                    "snap_time": 14.0,
                },
            ),
        ],
    },
    "plots/plate_reader_yfp_time_series": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="plot_set",
            summary="YFP plate reader time-series + distributions plot set.",
            tags=("plot", "yfp", "distribution"),
        ),
        "steps": [
            recipe_step(
                id="plot_time_series",
                plugin="plot/time_series",
                with_={
                    "partition": {"by": "design_id"},
                    "hue": "treatment",
                    "y": ["OD600", "YFP", "YFP/CFP", "YFP/OD600"],
                    "add_sheet_line": True,
                },
            ),
            recipe_step(
                id="distributions__by_design_id",
                plugin="plot/distributions",
                with_={"channels": ["YFP/CFP"], "partition": {"by": "design_id"}},
            ),
        ],
    },
    "plots/plate_reader_yfp_snapshots": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="plot_set",
            summary="YFP plate reader snapshot barplots.",
            tags=("plot", "yfp", "snapshot"),
        ),
        "steps": [
            recipe_step(
                id="snapshot_bars_by_channel",
                plugin="plot/snapshot_barplot",
                with_={"x": "treatment", "y": ["OD600", "YFP/OD600"], "partition": {"by": "design_id"}, "time": 14.0},
            ),
            recipe_step(
                id="snapshot_bars_by_design_id",
                plugin="plot/snapshot_barplot",
                with_={"x": "design_id", "y": "YFP/OD600", "hue": "treatment", "time": 14.0},
            ),
        ],
    },
    "plots/plate_reader_rfp_full": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="plot_set",
            summary="RFP plate reader plot set (time series + snapshots + TS+snap).",
            tags=("plot", "rfp"),
        ),
        "steps": [
            recipe_step(
                id="plot_time_series",
                plugin="plot/time_series",
                with_={
                    "partition": {"by": "design_id"},
                    "hue": "treatment",
                    "y": ["OD600", "RFP", "RFP/OD600"],
                    "add_sheet_line": True,
                },
            ),
            recipe_step(
                id="snapshot_bars_by_channel",
                plugin="plot/snapshot_barplot",
                with_={"x": "treatment", "y": ["OD600", "RFP/OD600"], "partition": {"by": "design_id"}, "time": 14.0},
            ),
            recipe_step(
                id="snapshot_bars_by_design_id",
                plugin="plot/snapshot_barplot",
                with_={"x": "design_id", "y": "RFP/OD600", "hue": "treatment", "time": 14.0},
            ),
            recipe_step(
                id="ts_and_snap__rfp_over_od600",
                plugin="plot/ts_and_snap",
                with_={
                    "partition": {"by": "design_id"},
                    "ts_channel": "RFP/OD600",
                    "ts_hue": "treatment",
                    "ts_add_sheet_line": True,
                    "ts_mark_snap_time": True,
                    "snap_channel": "RFP/OD600",
                    "snap_time": 14.0,
                },
            ),
        ],
    },
    "plots/plate_reader_rfp_time_series": {
        "semantics": WorkbenchRecipeSemantics(
            kind="recipe",
            domain="plate_reader",
            family="plot_set",
            summary="RFP plate reader time-series + snapshot barplots.",
            tags=("plot", "rfp", "snapshot"),
        ),
        "steps": [
            recipe_step(
                id="plot_time_series",
                plugin="plot/time_series",
                with_={
                    "partition": {"by": "design_id"},
                    "hue": "treatment",
                    "y": ["OD600", "RFP", "RFP/OD600"],
                    "add_sheet_line": True,
                },
            ),
            recipe_step(
                id="snapshot_bars_by_channel",
                plugin="plot/snapshot_barplot",
                reads={"df": {"record": "ratio_rfp_od600/df"}},
                with_={"x": "treatment", "y": ["OD600", "RFP/OD600"], "partition": {"by": "design_id"}, "time": 14.0},
            ),
            recipe_step(
                id="snapshot_bars_by_design_id",
                plugin="plot/snapshot_barplot",
                reads={"df": {"record": "ratio_rfp_od600/df"}},
                with_={"x": "design_id", "y": "RFP/OD600", "hue": "treatment", "time": 14.0},
            ),
        ],
    },
}
