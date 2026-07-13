"""
SFXI setpoint scatter plot plugin.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from reader.errors import SFXIError
from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig, PreflightIssue


class SFXISetpointScatterCfg(PluginConfig):
    setpoints: dict[str, list[float]] = Field(default_factory=lambda: {"and": [0.0, 0.0, 0.0, 1.0]})
    scaling_percentile: int = 95
    scaling_min_n: int = 5
    scaling_eps: float = 1.0e-8
    logic_exponent_beta: float = 1.0
    intensity_exponent_gamma: float = 1.0
    intensity_log2_offset_delta: float = 0.0
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None
    format: list[str] = Field(default_factory=lambda: ["pdf"])
    dpi: int = 300
    label_points: bool = False


class SFXISetpointScatterPlot(FigurePlotPlugin):
    ConfigModel = SFXISetpointScatterCfg

    @classmethod
    def input_ports(cls):
        return {"vec8": dataframe_input("vec8", "sfxi.vec8.v3")}

    @classmethod
    def preflight_readiness(cls, *, exp_dir, cfg: SFXISetpointScatterCfg, reads):
        del exp_dir, cfg, reads
        from reader.domains.logic.sfxi.setpoint_scatter import require_dnadesign_sfxi_api  # noqa: PLC0415

        try:
            require_dnadesign_sfxi_api()
        except SFXIError as exc:
            return (PreflightIssue(kind="dependency", message=str(exc)),)
        return ()

    def render(self, ctx, inputs, cfg: SFXISetpointScatterCfg) -> list[PlotFigure]:
        vec8: pd.DataFrame = inputs["vec8"]
        from reader.domains.logic.sfxi.setpoint_scatter import render_sfxi_setpoint_scatter  # noqa: PLC0415

        return render_sfxi_setpoint_scatter(
            vec8=vec8,
            setpoints=cfg.setpoints,
            scaling_percentile=cfg.scaling_percentile,
            scaling_min_n=cfg.scaling_min_n,
            scaling_eps=cfg.scaling_eps,
            logic_exponent_beta=cfg.logic_exponent_beta,
            intensity_exponent_gamma=cfg.intensity_exponent_gamma,
            intensity_log2_offset_delta=cfg.intensity_log2_offset_delta,
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            formats=cfg.format,
            dpi=cfg.dpi,
            label_points=cfg.label_points,
        )
