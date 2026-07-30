from __future__ import annotations

from typing import Literal

from pydantic import Field

from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


class ResponseWindowDiagnosticCfg(PluginConfig):
    source_experiment_id: str = Field(min_length=1)
    design_id: str = Field(min_length=1)
    primary_reduction_id: str = Field(min_length=1)
    pre_window_duration_h: float | None = Field(default=None, gt=0.0)
    title: str | None = Field(default=None, min_length=1)
    filename: str = Field(default="response_window_diagnostic", min_length=1)
    format: list[Literal["png", "pdf", "svg"]] = Field(default_factory=lambda: ["png"], min_length=1)
    dpi: int = Field(default=300, ge=1)


class ResponseWindowDiagnosticPlot(FigurePlotPlugin):
    ConfigModel = ResponseWindowDiagnosticCfg

    @classmethod
    def input_ports(cls):
        return {
            "traces": dataframe_input("traces", "plate_reader.response_window.traces.v3"),
            "designs": dataframe_input("designs", "plate_reader.response_window.designs.v4"),
        }

    def render(self, ctx, inputs, cfg: ResponseWindowDiagnosticCfg):
        from reader_workbench.domains.plate_reader.plots.response_window.diagnostic_render import (  # noqa: PLC0415
            render_response_window_diagnostic,
        )

        figure = render_response_window_diagnostic(
            inputs["traces"],
            inputs["designs"],
            source_experiment_id=cfg.source_experiment_id,
            design_id=cfg.design_id,
            reduction_id=cfg.primary_reduction_id,
            pre_window_duration_h=cfg.pre_window_duration_h,
            title=cfg.title,
        )
        return [
            PlotFigure(
                fig=figure,
                filename=cfg.filename,
                ext=extension,
                dpi=cfg.dpi,
                description=(
                    "Event-relative growth, response, magnitude, and reduced components "
                    "for one source experiment and design."
                ),
            )
            for extension in cfg.format
        ]
