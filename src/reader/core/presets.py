"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/presets.py

Built-in pipeline presets for common experiment workflows.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from reader.core.errors import ConfigError

Step = dict[str, Any]


def _rc_base() -> dict[str, Any]:
    return {
        "figure_figsize": [4, 4],
        "savefig_dpi": 300,
        "pdf_compression": 9,
        "font_scale": 1.10,
        "legend_fontsize": 10.0,
        "legend_title_fontsize": 10.0,
    }


def _rc_snap() -> dict[str, Any]:
    return {
        "figure_figsize": [4, 4],
        "savefig_dpi": 300,
        "pdf_compression": 9,
        "font_scale": 1.20,
    }


def _rc_ts_snap() -> dict[str, Any]:
    return {
        "figure_figsize": [8, 4],
        "savefig_dpi": 300,
        "pdf_compression": 9,
        "font_scale": 1.15,
        "legend_fontsize": 10.0,
        "legend_title_fontsize": 10.0,
    }


def _plot_steps_yfp(*, include_time_series: bool, include_ts_and_snap: bool) -> list[Step]:
    steps: list[Step] = [
        {
            "id": "distributions__by_design_id",
            "uses": "plot/distributions",
            "reads": {"df": "ratio_yfp_od600/df", "blanks": "blank/blanks"},
            "with": {
                "channels": ["OD600", "CFP", "YFP", "CFP/OD600", "YFP/OD600", "YFP/CFP"],
                "group_on": "design_id",
                "panel_by": "channel",
                "hue": "treatment_alias",
                "legend_loc": "upper left",
                "fig": {"rc": _rc_base(), "ext": "pdf"},
            },
        }
    ]
    if include_time_series:
        steps.append(
            {
                "id": "plot_time_series",
                "uses": "plot/time_series",
                "reads": {"df": "ratio_yfp_od600/df"},
                "with": {
                    "x": "time",
                    "y": ["OD600", "CFP", "YFP", "YFP/CFP", "CFP/OD600", "YFP/OD600"],
                    "hue": "treatment_alias",
                    "group_on": "design_id",
                    "pool_sets": None,
                    "ci": 95,
                    "ci_alpha": 0.15,
                    "add_sheet_line": True,
                    "sheet_line_kwargs": {"linestyle": "--", "color": "#9E9E9E", "linewidth": 0.8, "alpha": 0.9},
                    "legend_loc": "upper left",
                    "fig": {"line_alpha": 0.85, "mean_marker_alpha": 0.70, "replicate_alpha": 0.30, "rc": _rc_base()},
                },
            }
        )
    steps.extend(
        [
            {
                "id": "snapshot_bars_by_channel",
                "uses": "plot/snapshot_barplot",
                "reads": {"df": "ratio_yfp_od600/df"},
                "with": {
                    "x": "treatment",
                    "y": ["OD600", "YFP/OD600", "YFP/CFP"],
                    "group_on": "design_id",
                    "pool_sets": None,
                    "panel_by": "channel",
                    "time": 12.0,
                    "hue": None,
                    "show_legend": False,
                    "fig": {"rc": _rc_snap()},
                },
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "reads": {"df": "ratio_yfp_od600/df"},
                "with": {
                    "x": "treatment",
                    "y": "YFP/CFP",
                    "group_on": "design_id",
                    "pool_sets": None,
                    "panel_by": "group",
                    "time": 12.0,
                    "hue": None,
                    "show_legend": False,
                    "fig": {"rc": _rc_base()},
                },
            },
        ]
    )
    if include_ts_and_snap:
        steps.append(
            {
                "id": "ts_and_snap__yfp_over_od600",
                "uses": "plot/ts_and_snap",
                "reads": {"df": "ratio_yfp_od600/df"},
                "with": {
                    "group_on": "design_id",
                    "pool_sets": None,
                    "pool_match": "exact",
                    "ts_x": "time",
                    "ts_channel": "OD600",
                    "ts_hue": "treatment_alias",
                    "ts_time_window": None,
                    "ts_add_sheet_line": True,
                    "ts_sheet_line_kwargs": {"linestyle": "--", "color": "#9E9E9E", "linewidth": 0.8, "alpha": 0.9},
                    "ts_log_transform": False,
                    "ts_ci": 95,
                    "ts_ci_alpha": 0.15,
                    "ts_show_replicates": False,
                    "ts_legend_loc": "upper left",
                    "ts_mark_snap_time": True,
                    "ts_snap_line_kwargs": {"linestyle": "--", "color": "#9E9E9E", "linewidth": 0.9, "alpha": 1.0},
                    "snap_x": "treatment",
                    "snap_channel": "YFP/OD600",
                    "snap_hue": None,
                    "snap_time": 12.0,
                    "snap_agg": "mean",
                    "snap_err": "sem",
                    "snap_time_tolerance": 0.51,
                    "snap_show_legend": False,
                    "snap_legend_loc": "upper left",
                    "filename": "ts+snap__",
                    "fig": {"rc": _rc_ts_snap(), "ext": "pdf"},
                },
            }
        )
    return steps


def _plot_steps_rfp(*, include_time_series: bool, include_ts_and_snap: bool) -> list[Step]:
    steps: list[Step] = [
        {
            "id": "distributions__by_design_id",
            "uses": "plot/distributions",
            "reads": {"df": "ratio_rfp_od600/df", "blanks": "blank/blanks"},
            "with": {
                "channels": ["OD600", "RFP", "RFP/OD600"],
                "group_on": "design_id",
                "panel_by": "channel",
                "hue": "treatment",
                "legend_loc": "upper left",
                "fig": {"rc": _rc_base(), "ext": "pdf"},
            },
        }
    ]
    if include_time_series:
        steps.append(
            {
                "id": "plot_time_series",
                "uses": "plot/time_series",
                "reads": {"df": "ratio_rfp_od600/df"},
                "with": {
                    "x": "time",
                    "y": ["OD600", "RFP", "RFP/OD600"],
                    "hue": "treatment",
                    "group_on": "design_id",
                    "pool_sets": None,
                    "ci": 95,
                    "ci_alpha": 0.15,
                    "add_sheet_line": True,
                    "sheet_line_kwargs": {"linestyle": "--", "color": "#9E9E9E", "linewidth": 0.8, "alpha": 0.9},
                    "legend_loc": "upper left",
                    "fig": {"line_alpha": 0.85, "mean_marker_alpha": 0.70, "replicate_alpha": 0.30, "rc": _rc_base()},
                },
            }
        )
    steps.extend(
        [
            {
                "id": "snapshot_bars_by_channel",
                "uses": "plot/snapshot_barplot",
                "reads": {"df": "ratio_rfp_od600/df"},
                "with": {
                    "x": "treatment",
                    "y": ["OD600", "RFP", "RFP/OD600"],
                    "group_on": "design_id",
                    "pool_sets": None,
                    "panel_by": "channel",
                    "time": 12.0,
                    "hue": None,
                    "show_legend": False,
                    "fig": {"rc": _rc_snap()},
                },
            },
            {
                "id": "snapshot_bars_by_design_id",
                "uses": "plot/snapshot_barplot",
                "reads": {"df": "ratio_rfp_od600/df"},
                "with": {
                    "x": "treatment",
                    "y": "RFP/OD600",
                    "group_on": "design_id",
                    "pool_sets": None,
                    "panel_by": "group",
                    "time": 12.0,
                    "hue": None,
                    "show_legend": False,
                    "fig": {"rc": _rc_base()},
                },
            },
        ]
    )
    if include_ts_and_snap:
        steps.append(
            {
                "id": "ts_and_snap__rfp_over_od600",
                "uses": "plot/ts_and_snap",
                "reads": {"df": "ratio_rfp_od600/df"},
                "with": {
                    "group_on": "design_id",
                    "pool_sets": None,
                    "pool_match": "exact",
                    "ts_x": "time",
                    "ts_channel": "OD600",
                    "ts_hue": "treatment",
                    "ts_time_window": None,
                    "ts_add_sheet_line": True,
                    "ts_sheet_line_kwargs": {"linestyle": "--", "color": "#9E9E9E", "linewidth": 0.8, "alpha": 0.9},
                    "ts_log_transform": False,
                    "ts_ci": 95,
                    "ts_ci_alpha": 0.15,
                    "ts_show_replicates": False,
                    "ts_legend_loc": "upper left",
                    "ts_mark_snap_time": True,
                    "ts_snap_line_kwargs": {"linestyle": "--", "color": "#9E9E9E", "linewidth": 0.9, "alpha": 1.0},
                    "snap_x": "treatment",
                    "snap_channel": "RFP/OD600",
                    "snap_hue": None,
                    "snap_time": 12.0,
                    "snap_agg": "mean",
                    "snap_err": "sem",
                    "snap_time_tolerance": 0.51,
                    "snap_show_legend": False,
                    "snap_legend_loc": "upper left",
                    "filename": "ts+snap__",
                    "fig": {"rc": _rc_ts_snap(), "ext": "pdf"},
                },
            }
        )
    return steps


PRESETS: dict[str, dict[str, Any]] = {
    "plate_reader/synergy_h1": {
        "description": "Synergy H1 ingest with standard discovery defaults (override mode/channels/sheets).",
        "steps": [
            {
                "id": "ingest",
                "uses": "ingest/synergy_h1",
                "with": {
                    "add_sheet": True,
                    "auto_roots": ["./inputs"],
                    "auto_include": ["*.xlsx", "*.xls"],
                    "auto_exclude": ["~$*", "._*", "#*#", "*.tmp"],
                    "auto_pick": "single",
                    "auto_recursive": False,
                    "print_summary": True,
                },
            }
        ],
    },
    "plate_reader/sample_map": {
        "description": "Merge metadata (sample map) into tidy data.",
        "steps": [
            {
                "id": "merge_map",
                "uses": "merge/sample_map",
                "reads": {"df": "ingest/df", "sample_map": "file:./metadata.xlsx"},
                "with": {"require_columns": ["treatment", "design_id"], "require_non_null": False},
            }
        ],
    },
    "plate_reader/basic": {
        "description": "Synergy H1 ingest → sample map → aliases → blank/overflow → YFP/CFP/OD600 ratios.",
        "steps": [
            {
                "id": "ingest",
                "uses": "ingest/synergy_h1",
                "with": {
                    "add_sheet": True,
                    "auto_roots": ["./inputs"],
                    "auto_include": ["*.xlsx", "*.xls"],
                    "auto_exclude": ["~$*", "._*", "#*#", "*.tmp"],
                    "auto_pick": "single",
                    "auto_recursive": False,
                    "print_summary": True,
                },
            },
            {
                "id": "merge_map",
                "uses": "merge/sample_map",
                "reads": {"df": "ingest/df", "sample_map": "file:./metadata.xlsx"},
                "with": {"require_columns": ["treatment", "design_id"], "require_non_null": False},
            },
            {
                "id": "aliases",
                "uses": "transform/alias",
                "reads": {"df": "merge_map/df"},
                "with": {"aliases": {"design_id": {}, "treatment": {}}, "in_place": False, "case_insensitive": True},
            },
            {
                "id": "blank",
                "uses": "transform/blank_correction",
                "reads": {"df": "aliases/df"},
                "with": {"method": "disregard"},
            },
            {
                "id": "overflow",
                "uses": "transform/overflow_handling",
                "reads": {"df": "blank/df"},
                "with": {"action": "max", "clip_quantile": 0.999},
            },
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
    "cytometer/basic": {
        "description": "Flow cytometer ingest → sample metadata merge (event-level tidy table).",
        "steps": [
            {
                "id": "ingest",
                "uses": "ingest/flow_cytometer",
                "with": {
                    "auto_roots": ["./inputs"],
                    "auto_include": ["*.fcs"],
                    "auto_exclude": ["~$*", "._*", "#*#", "*.tmp"],
                    "auto_pick": "merge",
                    "auto_recursive": False,
                    "channel_name_field": "pns",
                    "include_source_file": True,
                    "print_summary": True,
                },
            },
            {
                "id": "merge_metadata",
                "uses": "merge/sample_metadata",
                "reads": {"df": "ingest/df", "metadata": "file:./metadata.xlsx"},
                "with": {"require_columns": ["design_id", "treatment"], "require_non_null": False},
            },
        ],
    },
    "plate_reader/blank_overflow": {
        "description": "Blank correction then overflow handling (assumes alias step; override reads if needed).",
        "steps": [
            {
                "id": "blank",
                "uses": "transform/blank_correction",
                "reads": {"df": "aliases/df"},
                "with": {"method": "disregard"},
            },
            {
                "id": "overflow",
                "uses": "transform/overflow_handling",
                "reads": {"df": "blank/df"},
                "with": {"action": "max", "clip_quantile": 0.999},
            },
        ],
    },
    "plate_reader/ratios_yfp_cfp_od600": {
        "description": "YFP/CFP + CFP/OD600 + YFP/OD600 ratios.",
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
        "description": "RFP/OD600 ratio.",
        "steps": [
            {
                "id": "ratio_rfp_od600",
                "uses": "transform/ratio",
                "reads": {"df": "overflow/df"},
                "with": {"name": "RFP/OD600", "numerator": "RFP", "denominator": "OD600"},
            }
        ],
    },
    "plots/plate_reader_yfp_full": {
        "description": "YFP/CFP/OD600 distributions + time series + snapshots + ts_and_snap.",
        "steps": _plot_steps_yfp(include_time_series=True, include_ts_and_snap=True),
    },
    "plots/plate_reader_yfp_time_series": {
        "description": "YFP/CFP/OD600 distributions + time series + snapshots.",
        "steps": _plot_steps_yfp(include_time_series=True, include_ts_and_snap=False),
    },
    "plots/plate_reader_yfp_snapshots": {
        "description": "YFP/CFP/OD600 distributions + snapshots (no time series).",
        "steps": _plot_steps_yfp(include_time_series=False, include_ts_and_snap=False),
    },
    "plots/plate_reader_rfp_full": {
        "description": "RFP/OD600 distributions + time series + snapshots + ts_and_snap.",
        "steps": _plot_steps_rfp(include_time_series=True, include_ts_and_snap=True),
    },
    "plots/plate_reader_rfp_time_series": {
        "description": "RFP/OD600 distributions + time series + snapshots.",
        "steps": _plot_steps_rfp(include_time_series=True, include_ts_and_snap=False),
    },
    "plots/plate_reader_rfp_snapshots": {
        "description": "RFP/OD600 distributions + snapshots (no time series).",
        "steps": _plot_steps_rfp(include_time_series=False, include_ts_and_snap=False),
    },
    "sfxi/promote": {
        "description": "Promote tidy.v1 -> tidy+map.v2 (strict column checks).",
        "steps": [
            {
                "id": "promote_to_tidy_plus_map",
                "uses": "validator/to_tidy_plus_map",
                "reads": {"df": "ratio_yfp_od600/df"},
                "with": {
                    "require_columns": ["treatment", "design_id"],
                    "require_non_null": True,
                    "drop_where_null_in": [],
                },
            }
        ],
    },
    "sfxi/vec8": {
        "description": "SFXI vec8 transform with standard parameters.",
        "steps": [
            {
                "id": "sfxi_vec8",
                "uses": "transform/sfxi",
                "reads": {"df": "promote_to_tidy_plus_map/df"},
                "with": {
                    "response": {"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
                    "design_by": ["design_id"],
                    "time_mode": "nearest",
                    "target_time_h": 12.0,
                    "time_tolerance_h": 0.25,
                    "treatment_map": {
                        "00": "EtOH_0_percent_0nM_cipro",
                        "10": "EtOH_3_percent_0nM_cipro",
                        "01": "EtOH_0_percent_100nM_cipro",
                        "11": "EtOH_3_percent_100nM_cipro",
                    },
                    "reference": {"design_id": "pDual-10", "stat": "mean"},
                    "treatment_case_sensitive": True,
                    "require_all_corners_per_design": True,
                    "eps_ratio": 1e-9,
                    "eps_range": 1e-12,
                    "eps_ref": 1e-9,
                    "eps_abs": 0.0,
                    "ref_add_alpha": 0.0,
                    "log2_offset_delta": 0.0,
                    "exclude_reference_from_output": True,
                    "carry_metadata": ["sequence", "id"],
                },
            }
        ],
    },
}


def list_presets() -> list[tuple[str, str]]:
    return sorted((name, info["description"]) for name, info in PRESETS.items())


def describe_preset(name: str) -> dict[str, Any]:
    info = _get_preset(name)
    return {"name": name, "description": info["description"], "steps": deepcopy(info["steps"])}


def expand_presets(names: Iterable[str]) -> list[Step]:
    steps: list[Step] = []
    for name in names:
        info = _get_preset(name)
        steps.extend(deepcopy(info["steps"]))
    return steps


def expand_steps(*, raw_steps: list[Step], presets: Iterable[str], overrides: dict[str, Any]) -> list[Step]:
    steps: list[Step] = []
    steps.extend(expand_presets(presets))
    for item in raw_steps:
        if isinstance(item, dict) and "preset" in item:
            if set(item.keys()) != {"preset"}:
                raise ConfigError("Preset entries may only contain the key 'preset'.")
            steps.extend(expand_presets([str(item["preset"])]))
        else:
            steps.append(item)
    _apply_overrides(steps, overrides or {})
    _assert_unique_step_ids(steps)
    return steps


def _get_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ConfigError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def _assert_unique_step_ids(steps: list[Step]) -> None:
    ids = [s.get("id") for s in steps]
    if None in ids:
        raise ConfigError("All steps must have an 'id'.")
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise ConfigError(f"Duplicate step ids found: {dupes}")


def _apply_overrides(steps: list[Step], overrides: dict[str, Any]) -> None:
    if not overrides:
        return
    if not isinstance(overrides, dict):
        raise ConfigError("overrides must be a mapping of step_id -> patch")
    by_id = {s["id"]: s for s in steps}
    for step_id, patch in overrides.items():
        if step_id not in by_id:
            raise ConfigError(f"override refers to unknown step id '{step_id}'")
        if not isinstance(patch, dict):
            raise ConfigError(f"override for step '{step_id}' must be a mapping")
        invalid = set(patch.keys()) - {"with", "reads"}
        if invalid:
            raise ConfigError(f"override for step '{step_id}' has invalid keys: {sorted(invalid)}")
        for key in ("with", "reads"):
            if key not in patch:
                continue
            base = by_id[step_id].get(key, {})
            if base is None:
                base = {}
            if not isinstance(base, dict) or not isinstance(patch[key], dict):
                raise ConfigError(f"override for step '{step_id}' -> {key} must be a mapping")
            by_id[step_id][key] = _deep_merge(dict(base), patch[key])


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(dict(base[k]), v)
        else:
            base[k] = v
    return base
