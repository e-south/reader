"""SFXI: setpoint_fidelity_x_intensity → vec8"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from reader.plugins.transform._sfxi import build_sfxi_plugin_result, log_sfxi_plugin_result
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig


class SFXIResponseBinding(PluginConfig):
    logic_channel: str = Field(min_length=1)
    intensity_channel: str = Field(min_length=1)


class SFXIReferenceBinding(PluginConfig):
    design_id: str = Field(min_length=1)
    stat: Literal["mean", "median"] = "mean"


class SFXICfg(PluginConfig):
    response: SFXIResponseBinding
    design_by: list[str] = Field(default_factory=lambda: ["design_id"])
    time_column: str = "time"
    treatment_column: str | None = None
    time_mode: Literal["nearest", "last_before", "first_after", "exact"] = "nearest"
    target_time_h: float | None = None
    time_tolerance_h: float | None = 0.5
    state_map_ref: str = Field(min_length=1)
    reference: SFXIReferenceBinding
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
    ConfigModel = SFXICfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "plate_reader.annotated.v1")}

    @classmethod
    def output_ports(cls):
        return {"vec8": dataframe_output("vec8", "sfxi.vec8.v3")}

    def run(self, ctx, inputs, cfg: SFXICfg):
        result = build_sfxi_plugin_result(ctx=ctx, df=inputs["df"], cfg=cfg)
        log_sfxi_plugin_result(ctx=ctx, result=result)
        return {"vec8": result.vec8}
