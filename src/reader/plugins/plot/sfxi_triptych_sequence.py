"""
SFXI triptych sequence bundle plot plugin.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from reader.errors import SFXIError
from reader.plugins.plot._shared import FigurePlotPlugin
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig, PreflightIssue


class _StrictModel(BaseModel):
    model_config = {"extra": "forbid"}


class SFXITriptychSequenceSourceCfg(_StrictModel):
    dataset: str
    root: str | None = None
    required_overlays: list[str] = Field(default_factory=list)
    id_column: str = "id"
    sequence_column: str = "sequence"
    label_column: str = "usr_label__primary"
    annotations_column: str = "densegen__used_tfbs_detail"
    adapter_kind: str = "densegen_tfbs"


class SFXITriptychChannelsCfg(_StrictModel):
    growth: str = "OD600"
    ratio: str = "YFP/CFP"
    snapshot: str = "YFP/CFP"


class SFXITriptychSequencePanelCfg(_StrictModel):
    profile: str = "promoter_compact_slide.v1"
    target_width_px: int = 2200
    target_height_px: int = 310
    vertical_anchor: str = "center"
    canvas_top_pad_px: int = 0
    style_overrides: dict[str, Any] = Field(default_factory=dict)


class SFXITriptychTreatmentCfg(_StrictModel):
    state: str
    label: str
    short_label: str
    color: str


class SFXITriptychSequenceCfg(PluginConfig):
    sequence_source: SFXITriptychSequenceSourceCfg
    bundle_id: str = "sfxi_triptych_sequence"
    design_col: str = "design_id"
    sequence_id_col: str = "id"
    sequence_col: str = "sequence"
    time_col: str = "time"
    treatment_col: str = "treatment_alias"
    snapshot_target_time_h: float = 12.0
    induction_time_h: float | None = 12.0
    time_tolerance_h: float = 0.51
    channels: SFXITriptychChannelsCfg = Field(default_factory=SFXITriptychChannelsCfg)
    sequence_panel: SFXITriptychSequencePanelCfg = Field(default_factory=SFXITriptychSequencePanelCfg)
    treatments: list[SFXITriptychTreatmentCfg] = Field(default_factory=list)
    time_series: dict[str, Any] = Field(default_factory=dict)
    axis_limits: dict[str, Any] = Field(default_factory=dict)
    movie_enabled: bool = False
    movie_fps: float = 0.85
    dpi: int = 220
    limit: int | None = None


class SFXITriptychSequencePlot(FigurePlotPlugin):
    ConfigModel = SFXITriptychSequenceCfg

    @classmethod
    def input_ports(cls):
        return {
            "vec8": dataframe_input("vec8", "sfxi.vec8.v2"),
            "assay": dataframe_input("assay", "plate_reader.annotated.v1"),
        }

    @classmethod
    def preflight_readiness(cls, *, exp_dir, cfg: SFXITriptychSequenceCfg, reads):
        del reads
        from reader.domains.logic.sfxi.triptych_sequence_dnadesign import (  # noqa: PLC0415
            require_dnadesign_sequence_panel_api,
            require_usr_sequence_dataset,
        )

        if not cfg.sequence_source.dataset.strip():
            return (
                PreflightIssue(kind="dependency", message="sfxi_triptych_sequence requires sequence_source.dataset."),
            )
        try:
            _baserender, usr = require_dnadesign_sequence_panel_api()
        except SFXIError as exc:
            return (PreflightIssue(kind="dependency", message=str(exc)),)
        try:
            require_usr_sequence_dataset(
                usr=usr,
                root=cfg.sequence_source.root,
                dataset_name=cfg.sequence_source.dataset,
                exp_dir=exp_dir,
            )
        except SFXIError as exc:
            return (PreflightIssue(kind="dependency", message=str(exc)),)
        return ()

    def render(self, ctx, inputs, cfg: SFXITriptychSequenceCfg):
        del ctx, inputs, cfg
        raise NotImplementedError("SFXITriptychSequencePlot uses run() to write an atomic file bundle.")

    def run(self, ctx, inputs, cfg: SFXITriptychSequenceCfg):
        vec8: pd.DataFrame = inputs["vec8"]
        assay: pd.DataFrame = inputs["assay"]
        from reader.domains.logic.sfxi.triptych_sequence import (  # noqa: PLC0415
            render_sfxi_triptych_sequence_bundle,
        )

        paths = render_sfxi_triptych_sequence_bundle(
            ctx=ctx,
            vec8=vec8,
            assay=assay,
            config=cfg.model_dump(mode="python"),
        )
        return {"artifacts": [str(path) for path in paths]}
