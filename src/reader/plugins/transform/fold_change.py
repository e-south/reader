"""
--------------------------------------------------------------------------------
<reader project>
plugins/transform/fold_change.py

Fold-change report:
  • Selects the nearest snapshot time(s) per group
  • Computes FC against explicit baselines (global or per-group overrides)
  • Emits a validated artifact (fold_change.v1) — no mutation of the main tidy df
  • Prints a concise, rich stdout summary (via logger) for quick inspection

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from reader.domains.plate_reader.analysis import compute_fold_change_table
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig

# ----------------------------- config model -----------------------------


class FoldChangeCfg(PluginConfig):
    # What to compute
    target: str  # e.g., "YFP/CFP" or "YFP/OD600"
    report_times: list[float]  # e.g., [8.0, 14.0]
    time_tolerance: float = 0.51  # nearest-time selection tolerance (h)
    agg: Literal["median", "mean"] = "median"  # replicate aggregator

    # Grouping and labels
    treatment_column: str = "treatment"  # we will prefer '<col>_alias' when present
    group_by: list[str] = Field(default_factory=lambda: ["design_id"])

    # Baseline policy
    use_global_baseline: bool = False
    global_baseline_value: str | None = None  # used when use_global_baseline==True
    # overrides: list of maps; any keys matching group_by columns define a match; each must
    # include 'baseline_value'. Example:
    #   - { design_id: "araBADp", baseline_value: "0 uM arabinose" }
    overrides: list[dict[str, Any]] = Field(default_factory=list)

    # Output columns (names)
    fc_column: str = "FC"
    log2fc_column: str = "log2FC"

    # Attach extra metadata columns if present (won't be required by contract, just carried through)
    attach_metadata: list[str] = Field(default_factory=lambda: ["batch"])


class FoldChange(Plugin):
    """Contract-driven transform that emits a fold_change.v1 table."""

    ConfigModel = FoldChangeCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {"table": dataframe_output("table", "fold_change.v1")}

    # ------------------------------- run --------------------------------

    def run(self, ctx, inputs, cfg: FoldChangeCfg):
        return {"table": compute_fold_change_table(ctx, inputs["df"], cfg)}
