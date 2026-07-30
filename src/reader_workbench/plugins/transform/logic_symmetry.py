from __future__ import annotations

from typing import Literal

from pydantic import Field

from reader_workbench.workbench.ports import dataframe_input, dataframe_output
from reader_workbench.workbench.registry import Plugin, PluginConfig


class LogicSymmetryPrepCfg(PluginConfig):
    enable: bool = False
    mode: Literal["first", "last", "median", "exact", "nearest"] = "last"
    target_time: float | None = None
    tolerance: float = Field(0.51, ge=0)
    align_corners: bool = False
    case_sensitive_treatments: bool | None = None
    time_column: str = "time"


class LogicSymmetryCfg(PluginConfig):
    response_channel: str = Field(min_length=1)
    design_by: list[str] = Field(default_factory=lambda: ["design_id"], min_length=1)
    batch_col: str = Field("batch", min_length=1)
    treatment_column: str | None = None
    state_map_ref: str = Field(min_length=1)
    observation_stat: Literal["mean", "median"] = "mean"
    prep: LogicSymmetryPrepCfg = Field(default_factory=LogicSymmetryPrepCfg)


class LogicSymmetryTransform(Plugin):
    """Materialize logic-symmetry metrics as a normal pipeline record."""

    ConfigModel = LogicSymmetryCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "plate_reader.annotated.v1")}

    @classmethod
    def output_ports(cls):
        return {"table": dataframe_output("table", "logic_symmetry.v1")}

    def run(self, ctx, inputs, cfg: LogicSymmetryCfg):
        if ctx.experiment is None:
            raise ValueError("logic_symmetry requires experiment semantics in the run context")
        state_space = ctx.experiment.annotations.resolve_ordered_state_space(ref=cfg.state_map_ref)
        if state_space.state_ids != ("00", "10", "01", "11"):
            raise ValueError("Logic-symmetry state space must declare exactly 00, 10, 01, 11 in that order")

        from reader_workbench.domains.logic.logic_symmetry import summarize_logic_symmetry  # noqa: PLC0415

        prep = cfg.prep.model_dump()
        if prep["case_sensitive_treatments"] is None:
            prep["case_sensitive_treatments"] = state_space.case_sensitive
        table = summarize_logic_symmetry(
            inputs["df"],
            response_channel=cfg.response_channel,
            design_by=cfg.design_by,
            batch_col=cfg.batch_col,
            treatment_column=cfg.treatment_column or state_space.column,
            treatment_map=dict(state_space.source_values),
            treatment_case_sensitive=state_space.case_sensitive,
            observation_stat=cfg.observation_stat,
            prep=prep,
        )
        return {"table": table}
