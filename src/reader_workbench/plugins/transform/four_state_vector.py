"""Four-state logic-intensity measurement vector."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from reader_workbench.plugins.transform._four_state_vector import (
    build_four_state_vector_plugin_result,
    log_four_state_vector_plugin_result,
)
from reader_workbench.workbench.ports import dataframe_input, dataframe_output
from reader_workbench.workbench.registry import Plugin, PluginConfig


class FourStateVectorResponseBinding(PluginConfig):
    logic_channel: str = Field(min_length=1)
    intensity_channel: str = Field(min_length=1)


class FourStateVectorReferenceBinding(PluginConfig):
    design_id: str = Field(min_length=1)
    observation_stat: Literal["mean", "median"] = "mean"


class FourStateVectorCfg(PluginConfig):
    response: FourStateVectorResponseBinding
    design_by: list[str] = Field(default_factory=lambda: ["design_id"])
    time_column: str = "time"
    treatment_column: str | None = None
    time_mode: Literal["nearest", "last_before", "first_after", "exact"] = "nearest"
    target_time_h: float | None = None
    time_tolerance_h: float | None = 0.5
    state_map_ref: str = Field(min_length=1)
    reference: FourStateVectorReferenceBinding
    require_all_corners_per_design: bool = True
    eps_ratio: float = 1e-9
    eps_range: float = 1e-12
    eps_ref: float = 1e-9
    eps_abs: float = 0.0
    ref_add_alpha: float = 0.0
    log2_offset_delta: float = 0.0
    exclude_reference_from_output: bool = True
    carry_metadata: list[str] = Field(default_factory=lambda: ["sequence", "id"])


class FourStateVectorTransform(Plugin):
    ConfigModel = FourStateVectorCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "plate_reader.annotated.v1")}

    @classmethod
    def output_ports(cls):
        return {"vector": dataframe_output("vector", "logic.four_state_vector.v1")}

    def run(self, ctx, inputs, cfg: FourStateVectorCfg):
        result = build_four_state_vector_plugin_result(ctx=ctx, df=inputs["df"], cfg=cfg)
        log_four_state_vector_plugin_result(ctx=ctx, result=result)
        return {"vector": result.vector}
