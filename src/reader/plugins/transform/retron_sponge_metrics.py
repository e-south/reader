from __future__ import annotations

from typing import Literal

from pydantic import Field

from reader.domains.plate_reader.analysis import compute_retron_sponge_metrics
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig


class SpongeStatesCfg(PluginConfig):
    uninduced_unstressed: str = "-IPTG/-stress"
    induced_unstressed: str = "+IPTG/-stress"
    uninduced_stressed: str = "-IPTG/+stress"
    induced_stressed: str = "+IPTG/+stress"


class PlateauCfg(PluginConfig):
    mode: Literal["full_post_stress", "control_plateau"] = "full_post_stress"
    slope_tolerance: float = 0.01
    min_intervals: int = 2


class RetronSpongeMetricsCfg(PluginConfig):
    measurement_channel: str = "YFP/CFP"
    growth_channel: str = "OD600"
    design_column: str = "design_id_alias"
    state_column: str = "treatment_alias"
    raw_treatment_column: str = "treatment"
    plate_column: str | None = "sheet_name"
    replicate_column: str = "position"
    sensor_column: str | None = None
    sponge_column: str | None = None
    genotype_column: str | None = None
    stress_condition_column: str | None = None
    relevant_stress_column: str | None = None
    expected_sign_column: str | None = None
    relevant_sensor_pair_column: str | None = None
    matched_control_group_column: str | None = None
    sponge_family_size_column: str | None = None
    design_separator: str = "/"
    control_name: str = "tetO"
    no_stress_label: str = "H2O"
    stress_time_zero_policy: Literal["explicit", "largest_gap_midpoint"] = "largest_gap_midpoint"
    stress_time_zero_h: float | None = None
    pre_reads: int = 3
    endpoint_reads: int = 3
    states: SpongeStatesCfg = Field(default_factory=SpongeStatesCfg)
    plateau: PlateauCfg = Field(default_factory=PlateauCfg)
    relevant_stress_map: dict[str, str] = Field(default_factory=dict)
    sensor_target_map: dict[str, list[str]] = Field(default_factory=dict)
    expected_sign_map: dict[str, int] = Field(default_factory=dict)


class RetronSpongeMetrics(Plugin):
    ConfigModel = RetronSpongeMetricsCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {
            "trace": dataframe_output("trace", "plate_reader.sponge_trace.v1"),
            "summary": dataframe_output("summary", "plate_reader.sponge_summary.v1"),
        }

    def run(self, ctx, inputs, cfg: RetronSpongeMetricsCfg):
        trace, summary = compute_retron_sponge_metrics(ctx, inputs["df"], cfg)
        return {"trace": trace, "summary": summary}
