"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/presets/plate_reader.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

PLATE_READER_PRESETS: dict[str, dict[str, Any]] = {
    "plate_reader/synergy_h1": {
        "description": "Synergy H1 ingest (auto-discovery).",
        "steps": [{"id": "ingest", "uses": "ingest/synergy_h1", "reads": {}}],
    },
    "plate_reader/dual_reporter_screen_base": {
        "description": (
            "Dual-reporter screen base workflow: sample-map merge, design/treatment aliases, "
            "blank correction, overflow handling, and CFP/YFP/OD600 ratios."
        ),
        "steps": [
            {
                "id": "merge_map",
                "uses": "merge/sample_map",
                "reads": {"df": "ingest/df", "sample_map": "resource:sample_map"},
            },
            {
                "id": "labels",
                "uses": "transform/assay_labels",
                "reads": {"df": "merge_map/df"},
            },
            {
                "id": "blank",
                "uses": "transform/blank_correction",
                "reads": {"df": "labels/df"},
            },
            {
                "id": "overflow",
                "uses": "transform/overflow_handling",
                "reads": {"df": "blank/df"},
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
    "plate_reader/sample_map": {
        "description": "Merge plate sample map from inputs/metadata.xlsx.",
        "steps": [
            {
                "id": "merge_map",
                "uses": "merge/sample_map",
                "reads": {"df": "ingest/df", "sample_map": "resource:sample_map"},
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
}
