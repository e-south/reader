"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/presets.py

Preset bundles for common pipelines.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from typing import Any

from reader.core.errors import ConfigError

_PRESETS: dict[str, dict[str, Any]] = {
    "plate_reader/synergy_h1": {
        "description": "Synergy H1 ingest (auto-discovery).",
        "steps": [
            {
                "id": "ingest",
                "uses": "ingest/synergy_h1",
                "reads": {},
            }
        ],
    },
    "plate_reader/sample_map": {
        "description": "Merge plate sample map from inputs/metadata.xlsx.",
        "steps": [
            {
                "id": "merge_map",
                "uses": "merge/sample_map",
                "reads": {
                    "df": "ingest/df",
                    "sample_map": "file:./inputs/metadata.xlsx",
                },
            }
        ],
    },
    "plate_reader/blank_overflow": {
        "description": "Blank correction + overflow handling (expects aliases/df; override reads if needed).",
        "steps": [
            {"id": "blank", "uses": "transform/blank_correction", "reads": {"df": "aliases/df"}},
            {"id": "overflow", "uses": "transform/overflow_handling", "reads": {"df": "blank/df"}},
        ],
    },
    "plate_reader/ratios_yfp_cfp_od600": {
        "description": "Append YFP/CFP, CFP/OD600, YFP/OD600 ratios.",
        "steps": [
            {
                "id": "ratio_yfp_cfp",
                "uses": "transform/ratio",
                "reads": {"df": "overflow/df"},
                "with": {"name": "YFP/CFP", "numerator": "YFP", "denominator": "CFP"},
            },
            {
                "id": "ratio_cfp_od600",
                "uses": "transform/ratio",
                "reads": {"df": "ratio_yfp_cfp/df"},
                "with": {"name": "CFP/OD600", "numerator": "CFP", "denominator": "OD600"},
            },
            {
                "id": "ratio_yfp_od600",
                "uses": "transform/ratio",
                "reads": {"df": "ratio_cfp_od600/df"},
                "with": {"name": "YFP/OD600", "numerator": "YFP", "denominator": "OD600"},
            },
        ],
    },
    "plate_reader/ratio_rfp_od600": {
        "description": "Append RFP/OD600 ratio.",
        "steps": [
            {
                "id": "ratio_rfp_od600",
                "uses": "transform/ratio",
                "reads": {"df": "overflow/df"},
                "with": {"name": "RFP/OD600", "numerator": "RFP", "denominator": "OD600"},
            }
        ],
    },
    "sfxi/promote": {
        "description": "Promote tidy table to tidy+map for SFXI.",
        "steps": [
            {
                "id": "promote_to_tidy_plus_map",
                "uses": "validator/to_tidy_plus_map",
                "reads": {"df": "ratio_yfp_od600/df"},
            }
        ],
    },
    "sfxi/vec8": {
        "description": "Compute SFXI vec8 labels from tidy+map.",
        "steps": [
            {
                "id": "sfxi_vec8",
                "uses": "transform/sfxi",
                "reads": {"df": "promote_to_tidy_plus_map/df"},
                "with": {
                    "response": {"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
                    "design_by": ["design_id"],
                    "treatment_map": {
                        "00": "EtOH 0%, 0 nM cipro",
                        "10": "EtOH 3%, 0 nM cipro",
                        "01": "EtOH 0%, 100 nM cipro",
                        "11": "EtOH 3%, 100 nM cipro",
                    },
                },
            }
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
                "with": {
                    "x": "treatment",
                    "y": ["OD600", "YFP/OD600"],
                    "group_on": "design_id",
                    "time": 14.0,
                },
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "with": {
                    "x": "design_id",
                    "y": "YFP/OD600",
                    "hue": "treatment",
                    "time": 14.0,
                },
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
                "with": {
                    "channels": ["YFP/CFP"],
                    "group_on": "design_id",
                },
            },
        ],
    },
    "plots/plate_reader_yfp_snapshots": {
        "description": "YFP plate reader snapshot barplots.",
        "steps": [
            {
                "id": "snapshot_bars_by_channel",
                "uses": "plot/snapshot_barplot",
                "with": {
                    "x": "treatment",
                    "y": ["OD600", "YFP/OD600"],
                    "group_on": "design_id",
                    "time": 14.0,
                },
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "with": {
                    "x": "design_id",
                    "y": "YFP/OD600",
                    "hue": "treatment",
                    "time": 14.0,
                },
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
                "with": {
                    "x": "treatment",
                    "y": ["OD600", "RFP/OD600"],
                    "group_on": "design_id",
                    "time": 14.0,
                },
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "with": {
                    "x": "design_id",
                    "y": "RFP/OD600",
                    "hue": "treatment",
                    "time": 14.0,
                },
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
                "with": {
                    "x": "treatment",
                    "y": ["OD600", "RFP/OD600"],
                    "group_on": "design_id",
                    "time": 14.0,
                },
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "reads": {"df": "ratio_rfp_od600/df"},
                "with": {
                    "x": "design_id",
                    "y": "RFP/OD600",
                    "hue": "treatment",
                    "time": 14.0,
                },
            },
        ],
    },
}


def _infer_category(steps: list[dict[str, Any]]) -> str:
    cats = {str(step.get("uses", "")).split("/", 1)[0] for step in steps if isinstance(step, dict)}
    cats.discard("")
    if not cats:
        return "pipeline"
    if cats == {"plot"}:
        return "plot"
    if cats == {"export"}:
        return "export"
    return "pipeline"


def list_presets(category: str | None = None) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for name, info in _PRESETS.items():
        steps = info.get("steps", [])
        cat = _infer_category(steps)
        if category and cat != category:
            continue
        items.append((name, info["description"]))
    return sorted(items)


def resolve_preset(name: str) -> list[dict[str, Any]]:
    if name not in _PRESETS:
        opts = ", ".join(sorted(_PRESETS))
        raise ConfigError(f"Unknown preset {name!r}. Available presets: {opts}")
    return copy.deepcopy(_PRESETS[name]["steps"])


def describe_preset(name: str) -> dict[str, Any]:
    if name not in _PRESETS:
        opts = ", ".join(sorted(_PRESETS))
        raise ConfigError(f"Unknown preset {name!r}. Available presets: {opts}")
    info = _PRESETS[name]
    steps = copy.deepcopy(info["steps"])
    return {
        "name": name,
        "description": info.get("description", ""),
        "category": _infer_category(steps),
        "steps": steps,
    }
