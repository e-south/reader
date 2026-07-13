"""
SFXI triptych sequence bundle plot plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from reader.plugins.plot._shared import FigurePlotPlugin
from reader.workbench.ports import dataframe_input, file_path_input
from reader.workbench.registry import PluginConfig, PreflightIssue


class _StrictModel(BaseModel):
    model_config = {"extra": "forbid"}


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
    bundle_id: str = "sfxi_triptych_sequence"
    design_col: str = "design_id"
    sequence_col: str = "sequence"
    time_col: str = "time"
    treatment_col: str = "treatment_alias"
    snapshot_target_time_h: float = 12.0
    acquisition_transition_time_h: float | None = None
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
            "vec8": dataframe_input("vec8", "sfxi.vec8.v3"),
            "assay": dataframe_input("assay", "plate_reader.annotated.v1"),
            "candidate_bindings": file_path_input("candidate_bindings"),
        }

    @classmethod
    def preflight_readiness(cls, *, exp_dir, cfg: SFXITriptychSequenceCfg, reads):
        del cfg
        from reader.domains.promoter.candidate_bindings import load_promoter_candidate_bindings  # noqa: PLC0415
        from reader.domains.promoter.sequence_panel import (  # noqa: PLC0415
            PromoterSequencePanelError,
            require_baserender_api,
        )

        try:
            require_baserender_api()
        except PromoterSequencePanelError as exc:
            return (PreflightIssue(kind="dependency", message=str(exc)),)
        binding_ref = reads.get("candidate_bindings")
        raw_path = getattr(binding_ref, "path", None)
        if raw_path is None:
            return (PreflightIssue(kind="dependency", message="candidate_bindings resource is not declared."),)
        path = Path(raw_path)
        if not path.is_absolute():
            path = exp_dir / path
        try:
            load_promoter_candidate_bindings(path)
        except (FileNotFoundError, ValueError) as exc:
            return (PreflightIssue(kind="dependency", message=str(exc)),)
        return ()

    def render(self, ctx, inputs, cfg: SFXITriptychSequenceCfg):
        del ctx, inputs, cfg
        raise NotImplementedError("SFXITriptychSequencePlot uses run() to write an atomic file bundle.")

    def run(self, ctx, inputs, cfg: SFXITriptychSequenceCfg):
        vec8: pd.DataFrame = inputs["vec8"]
        assay: pd.DataFrame = inputs["assay"]
        candidate_bindings_manifest: Path = inputs["candidate_bindings"]
        from reader.domains.logic.sfxi.triptych_sequence import (  # noqa: PLC0415
            render_sfxi_triptych_sequence_bundle,
        )

        paths = render_sfxi_triptych_sequence_bundle(
            ctx=ctx,
            vec8=vec8,
            assay=assay,
            candidate_bindings_manifest=candidate_bindings_manifest,
            config=cfg.model_dump(mode="python"),
        )
        return {"artifacts": [str(path) for path in paths]}
