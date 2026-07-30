"""Static cytometry gating diagnostic plot plugin."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


class CytometryDiagnosticCfg(PluginConfig):
    title: str | None = Field(default=None, min_length=1)
    max_events: int = Field(default=50_000, gt=0)
    filename: str = Field(default="cytometry_diagnostic", min_length=1)
    format: list[Literal["png", "pdf", "svg"]] = Field(default_factory=lambda: ["png"], min_length=1)
    dpi: int = Field(default=300, gt=0)


class CytometryDiagnosticPlot(FigurePlotPlugin):
    ConfigModel = CytometryDiagnosticCfg

    @classmethod
    def input_ports(cls):
        return {
            "original_events": dataframe_input("original_events", "tidy.v1"),
            "gate_definition": dataframe_input("gate_definition", "cytometry.gate_definition.v1"),
            "gated_events": dataframe_input("gated_events", "cytometry.gated_events.v1"),
        }

    def render(self, ctx, inputs, cfg: CytometryDiagnosticCfg):
        del ctx
        from reader_workbench.domains.cytometry.plots import render_cytometry_diagnostic  # noqa: PLC0415

        figure = render_cytometry_diagnostic(
            inputs["original_events"],
            inputs["gate_definition"],
            inputs["gated_events"],
            max_events=cfg.max_events,
            title=cfg.title,
        )
        return [
            PlotFigure(
                fig=figure,
                filename=cfg.filename,
                ext=extension,
                dpi=cfg.dpi,
                description="Configured cells, singlets, fluorescence, and final-retention diagnostics.",
            )
            for extension in cfg.format
        ]
