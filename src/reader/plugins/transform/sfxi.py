"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/sfxi.py

SFXI: setpoint_fidelity_x_intensity → vec8

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field

from reader.core.registry import Plugin, PluginConfig
from reader.core.workbench import PluginSemantics
from reader.plugins.transform._sfxi import build_sfxi_plugin_result, log_sfxi_plugin_result


class SFXICfg(PluginConfig):
    response: dict[str, str]  # {"logic_channel":..., "intensity_channel":...}
    design_by: list[str] = Field(default_factory=lambda: ["design_id"])
    time_column: str = "time"
    time_mode: str = "nearest"  # nearest|last_before|first_after|exact
    target_time_h: float | None = None
    time_tolerance_h: float = 0.5
    logic_map_ref: str
    reference: dict[str, str | None] = Field(default_factory=lambda: {"design_id": None, "stat": "mean"})
    require_all_corners_per_design: bool = True
    eps_ratio: float = 1e-9
    eps_range: float = 1e-12
    eps_ref: float = 1e-9
    eps_abs: float = 0.0
    ref_add_alpha: float = 0.0
    log2_offset_delta: float = 0.0
    exclude_reference_from_output: bool = True
    carry_metadata: list[str] = Field(default_factory=lambda: ["sequence", "id"])


class SFXITransform(Plugin):
    key = "sfxi"
    category = "transform"
    semantics = PluginSemantics(
        category="transform",
        domain="logic",
        family="summary_transform",
        summary="Compute SFXI vec8 logic summaries from annotated plate-reader traces.",
        tags=("logic", "summary"),
    )
    ConfigModel = SFXICfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "plate_reader.annotated.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"vec8": "sfxi.vec8.v2"}

    def run(self, ctx, inputs, cfg: SFXICfg):
        result = build_sfxi_plugin_result(ctx=ctx, df=inputs["df"], cfg=cfg)
        log_sfxi_plugin_result(ctx=ctx, result=result)
        return {"vec8": result.vec8}
