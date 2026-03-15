"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/presets/plots.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

PLOT_PRESETS: dict[str, dict[str, Any]] = {
    "plots/plate_reader_dual_reporter_screen_core": {
        "description": "Core plot set for dual-reporter plate-reader screens.",
        "steps": [
            {
                "id": "plot_time_series",
                "uses": "plot/time_series",
                "with": {
                    "group_on": "design_id_alias",
                    "hue": "treatment_alias",
                    "y": ["OD600", "CFP", "YFP", "YFP/CFP", "CFP/OD600", "YFP/OD600"],
                    "add_sheet_line": True,
                },
            },
            {
                "id": "snapshot_bars_by_state",
                "uses": "plot/snapshot_barplot",
                "with": {
                    "x": "treatment_alias",
                    "y": ["OD600", "CFP/OD600", "YFP/OD600", "YFP/CFP"],
                    "group_on": "design_id_alias",
                    "time": 14.0,
                },
            },
            {
                "id": "ts_and_snap__yfp_over_cfp",
                "uses": "plot/ts_and_snap",
                "with": {
                    "group_on": "design_id_alias",
                    "ts_channel": "OD600",
                    "ts_hue": "treatment_alias",
                    "ts_add_sheet_line": True,
                    "ts_mark_snap_time": True,
                    "snap_x": "treatment_alias",
                    "snap_channel": "YFP/CFP",
                    "snap_time": 14.0,
                },
            },
        ],
    },
    "plots/plate_reader_yfp_full": {
        "description": "YFP plate reader plot set (time series + snapshots + TS+snap).",
        "steps": [
            {
                "id": "plot_time_series",
                "uses": "plot/time_series",
                "with": {
                    "group_on": "design_id",
                    "hue": "treatment",
                    "y": ["OD600", "YFP", "YFP/CFP", "YFP/OD600"],
                    "add_sheet_line": True,
                },
            },
            {
                "id": "snapshot_bars_by_channel",
                "uses": "plot/snapshot_barplot",
                "with": {"x": "treatment", "y": ["OD600", "YFP/OD600"], "group_on": "design_id", "time": 14.0},
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "with": {"x": "design_id", "y": "YFP/OD600", "hue": "treatment", "time": 14.0},
            },
            {
                "id": "ts_and_snap__yfp_over_od600",
                "uses": "plot/ts_and_snap",
                "with": {
                    "group_on": "design_id",
                    "ts_channel": "YFP/OD600",
                    "ts_hue": "treatment",
                    "ts_add_sheet_line": True,
                    "ts_mark_snap_time": True,
                    "snap_channel": "YFP/OD600",
                    "snap_time": 14.0,
                },
            },
        ],
    },
    "plots/plate_reader_yfp_time_series": {
        "description": "YFP plate reader time-series + distributions plot set.",
        "steps": [
            {
                "id": "plot_time_series",
                "uses": "plot/time_series",
                "with": {
                    "group_on": "design_id",
                    "hue": "treatment",
                    "y": ["OD600", "YFP", "YFP/CFP", "YFP/OD600"],
                    "add_sheet_line": True,
                },
            },
            {
                "id": "distributions__by_design_id",
                "uses": "plot/distributions",
                "with": {"channels": ["YFP/CFP"], "group_on": "design_id"},
            },
        ],
    },
    "plots/plate_reader_yfp_snapshots": {
        "description": "YFP plate reader snapshot barplots.",
        "steps": [
            {
                "id": "snapshot_bars_by_channel",
                "uses": "plot/snapshot_barplot",
                "with": {"x": "treatment", "y": ["OD600", "YFP/OD600"], "group_on": "design_id", "time": 14.0},
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "with": {"x": "design_id", "y": "YFP/OD600", "hue": "treatment", "time": 14.0},
            },
        ],
    },
    "plots/plate_reader_rfp_full": {
        "description": "RFP plate reader plot set (time series + snapshots + TS+snap).",
        "steps": [
            {
                "id": "plot_time_series",
                "uses": "plot/time_series",
                "with": {
                    "group_on": "design_id",
                    "hue": "treatment",
                    "y": ["OD600", "RFP", "RFP/OD600"],
                    "add_sheet_line": True,
                },
            },
            {
                "id": "snapshot_bars_by_channel",
                "uses": "plot/snapshot_barplot",
                "with": {"x": "treatment", "y": ["OD600", "RFP/OD600"], "group_on": "design_id", "time": 14.0},
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "with": {"x": "design_id", "y": "RFP/OD600", "hue": "treatment", "time": 14.0},
            },
            {
                "id": "ts_and_snap__rfp_over_od600",
                "uses": "plot/ts_and_snap",
                "with": {
                    "group_on": "design_id",
                    "ts_channel": "RFP/OD600",
                    "ts_hue": "treatment",
                    "ts_add_sheet_line": True,
                    "ts_mark_snap_time": True,
                    "snap_channel": "RFP/OD600",
                    "snap_time": 14.0,
                },
            },
        ],
    },
    "plots/plate_reader_rfp_time_series": {
        "description": "RFP plate reader time-series + snapshot barplots.",
        "steps": [
            {
                "id": "plot_time_series",
                "uses": "plot/time_series",
                "with": {
                    "group_on": "design_id",
                    "hue": "treatment",
                    "y": ["OD600", "RFP", "RFP/OD600"],
                    "add_sheet_line": True,
                },
            },
            {
                "id": "snapshot_bars_by_channel",
                "uses": "plot/snapshot_barplot",
                "reads": {"df": "ratio_rfp_od600/df"},
                "with": {"x": "treatment", "y": ["OD600", "RFP/OD600"], "group_on": "design_id", "time": 14.0},
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "reads": {"df": "ratio_rfp_od600/df"},
                "with": {"x": "design_id", "y": "RFP/OD600", "hue": "treatment", "time": 14.0},
            },
        ],
    },
}
