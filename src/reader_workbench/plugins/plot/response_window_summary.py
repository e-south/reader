from __future__ import annotations

from typing import Literal

from pydantic import Field

from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


class ResponseWindowSummaryCfg(PluginConfig):
    primary_reduction_id: str = Field(min_length=1)
    experiment_ids: list[str] | None = Field(default=None, min_length=1)
    design_ids: list[str] | None = Field(default=None, min_length=1)
    maximum_rows: int = Field(default=64, ge=1)
    title: str = Field(default="Response-window summary", min_length=1)
    filename: str = Field(default="response_window_summary", min_length=1)
    format: list[Literal["png", "pdf", "svg"]] = Field(default_factory=lambda: ["png"], min_length=1)
    dpi: int = Field(default=300, ge=1)


class ResponseWindowSummaryPlot(FigurePlotPlugin):
    ConfigModel = ResponseWindowSummaryCfg

    @classmethod
    def input_ports(cls):
        return {"designs": dataframe_input("designs", "plate_reader.response_window.designs.v4")}

    def render(self, ctx, inputs, cfg):
        from reader_workbench.domains.plate_reader.plots.response_window.summary import (  # noqa: PLC0415
            render_response_window_summary,
        )

        figure = render_response_window_summary(
            inputs["designs"],
            primary_reduction_id=cfg.primary_reduction_id,
            experiment_ids=cfg.experiment_ids,
            design_ids=cfg.design_ids,
            maximum_rows=cfg.maximum_rows,
            title=cfg.title,
        )
        return [
            PlotFigure(
                fig=figure,
                filename=cfg.filename,
                ext=extension,
                dpi=cfg.dpi,
                description="Primary event-relative response and anchored-magnitude components by source and design.",
            )
            for extension in cfg.format
        ]
