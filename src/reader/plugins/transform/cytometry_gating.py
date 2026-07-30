"""Explicit cytometry gating transform."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from reader.domains.cytometry.analysis import GateSpec, ThresholdSpec
from reader.domains.cytometry.analysis.workflow import CytometryGatingRequest, CytometryQCSpec, run_cytometry_gating
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig


class CytometryGatingCfg(PluginConfig):
    cells_x_channel: str = Field(min_length=1)
    cells_y_channel: str = Field(min_length=1)
    cells_x_range: tuple[float, float]
    cells_y_range: tuple[float, float]
    singlet_x_channel: str = Field(min_length=1)
    singlet_y_channel: str = Field(min_length=1)
    singlet_ratio_range: tuple[float, float]
    cells_enabled: bool
    singlets_enabled: bool
    fluorescence_channel: str = Field(min_length=1)
    threshold_mode: Literal["manual", "from_control_quantile"]
    threshold_value: float | None
    threshold_group_column: str | None
    threshold_control_value: str | None
    threshold_quantile: float | None
    group_column: str | None
    minimum_final_events: int = Field(ge=0)
    minimum_final_percent: float = Field(ge=0.0, le=100.0)
    maximum_nonpositive_percent: float = Field(ge=0.0, le=100.0)
    nonpositive_scope: Literal["all_events", "gated_events"]

    @model_validator(mode="after")
    def _validate_threshold_policy(self):
        if self.threshold_mode == "manual":
            if self.threshold_value is None:
                raise ValueError("threshold_value is required for manual thresholding")
            if any(
                value is not None
                for value in (self.threshold_group_column, self.threshold_control_value, self.threshold_quantile)
            ):
                raise ValueError("manual thresholding may not declare control threshold fields")
            return self
        if self.threshold_value is not None:
            raise ValueError("threshold_value must be null for control-quantile thresholding")
        if not self.threshold_group_column:
            raise ValueError("threshold_group_column is required for control-quantile thresholding")
        if not self.threshold_control_value:
            raise ValueError("threshold_control_value is required for control-quantile thresholding")
        if self.threshold_quantile is None or not 0.0 <= self.threshold_quantile <= 1.0:
            raise ValueError("threshold_quantile must be between 0 and 1 for control-quantile thresholding")
        return self


class CytometryGatingTransform(Plugin):
    ConfigModel = CytometryGatingCfg

    @classmethod
    def input_ports(cls):
        return {"events": dataframe_input("events", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {
            "gate_definition": dataframe_output("gate_definition", "cytometry.gate_definition.v1"),
            "gated_events": dataframe_output("gated_events", "cytometry.gated_events.v1"),
            "sample_stats": dataframe_output("sample_stats", "cytometry.sample_stats.v1"),
            "group_stats": dataframe_output("group_stats", "cytometry.group_stats.v1"),
            "qc": dataframe_output("qc", "cytometry.qc.v1"),
        }

    def run(self, ctx, inputs, cfg: CytometryGatingCfg):
        del ctx
        threshold = ThresholdSpec(
            channel=cfg.fluorescence_channel,
            value=float(cfg.threshold_value or 0.0),
            mode=cfg.threshold_mode,
            group_column=cfg.threshold_group_column,
            control_value=cfg.threshold_control_value,
            quantile=float(cfg.threshold_quantile or 0.0),
        )
        result = run_cytometry_gating(
            inputs["events"],
            CytometryGatingRequest(
                gate=GateSpec(
                    cells_x_channel=cfg.cells_x_channel,
                    cells_y_channel=cfg.cells_y_channel,
                    cells_x_range=cfg.cells_x_range,
                    cells_y_range=cfg.cells_y_range,
                    singlet_x_channel=cfg.singlet_x_channel,
                    singlet_y_channel=cfg.singlet_y_channel,
                    singlet_ratio_range=cfg.singlet_ratio_range,
                    cells_enabled=cfg.cells_enabled,
                    singlets_enabled=cfg.singlets_enabled,
                ),
                threshold=threshold,
                group_column=cfg.group_column,
                qc=CytometryQCSpec(
                    minimum_final_events=cfg.minimum_final_events,
                    minimum_final_percent=cfg.minimum_final_percent,
                    maximum_nonpositive_percent=cfg.maximum_nonpositive_percent,
                    nonpositive_scope=cfg.nonpositive_scope,
                ),
            ),
        )
        return {
            "gate_definition": result.gate_definition.to_pandas(),
            "gated_events": result.gated_events.to_pandas(),
            "sample_stats": result.sample_stats.to_pandas(),
            "group_stats": result.group_stats.to_pandas(),
            "qc": result.qc.to_pandas(),
        }
