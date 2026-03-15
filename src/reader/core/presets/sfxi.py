"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/presets/sfxi.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

SFXI_PRESETS: dict[str, dict[str, Any]] = {
    "sfxi/promote": {
        "description": "Promote tidy table to annotated plate-reader data for SFXI.",
        "steps": [
            {
                "id": "promote_to_tidy_plus_map",
                "uses": "validator/to_tidy_plus_map",
                "reads": {"df": "ratio_yfp_od600/df"},
                "with": {"synthesize_batch": True},
            }
        ],
    },
    "sfxi/vec8": {
        "description": "Compute SFXI vec8 labels from annotated plate-reader data.",
        "steps": [
            {
                "id": "sfxi_vec8",
                "uses": "transform/sfxi",
                "reads": {"df": "promote_to_tidy_plus_map/df"},
                "with": {
                    "response": {"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
                    "design_by": ["design_id"],
                    "logic_map_ref": "induction_logic",
                },
            }
        ],
    },
}
