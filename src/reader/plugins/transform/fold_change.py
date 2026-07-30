"""Fold-change report:
• Selects the nearest snapshot time(s) per group
• Computes FC against explicit baselines (global or per-group overrides)
• Emits a validated artifact (fold_change.v1) — no mutation of the main tidy df
• Prints a concise, rich stdout summary (via logger) for quick inspection"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from reader.domains.plate_reader.analysis import FoldChangeAnalysisSpec, compute_fold_change_table
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig

# ----------------------------- config model -----------------------------


class FoldChangeCfg(PluginConfig):
    # What to compute
    target: str  # e.g., "YFP/CFP" or "YFP/OD600"
    report_times: list[float]  # e.g., [8.0, 14.0]
    time_tolerance: float = 0.51  # nearest-time selection tolerance (h)
    observation_stat: Literal["median", "mean"] = "median"

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
        spec = FoldChangeAnalysisSpec(
            target=cfg.target,
            report_times=tuple(cfg.report_times),
            time_tolerance=cfg.time_tolerance,
            observation_stat=cfg.observation_stat,
            treatment_column=cfg.treatment_column,
            group_by=tuple(cfg.group_by),
            use_global_baseline=cfg.use_global_baseline,
            global_baseline_value=cfg.global_baseline_value,
            overrides=tuple(dict(rule) for rule in cfg.overrides),
            fc_column=cfg.fc_column,
            log2fc_column=cfg.log2fc_column,
            attach_metadata=tuple(cfg.attach_metadata),
        )
        return {"table": compute_fold_change_table(inputs["df"], spec=spec, logger=ctx.logger)}
