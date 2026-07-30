from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


class LogicSymmetryPlotCfg(PluginConfig):
    title: str = "Logic symmetry"
    dispersion: Literal["none", "bars", "halo"] = "halo"
    encodings: dict[str, Any] = Field(default_factory=dict)
    ideals_overlay: dict[str, Any] = Field(default_factory=dict)
    visuals: dict[str, Any] = Field(default_factory=dict)
    filename: str = "logic_symmetry"
    format: list[Literal["png", "pdf", "svg"]] = Field(default_factory=lambda: ["pdf"], min_length=1)
    dpi: int = Field(300, gt=0)
    figsize: tuple[float, float] = (7, 6)


class LogicSymmetryPlot(FigurePlotPlugin):
    """Render a persisted logic-symmetry table without owning its computation."""

    ConfigModel = LogicSymmetryPlotCfg

    @classmethod
    def input_ports(cls):
        return {"table": dataframe_input("table", "logic_symmetry.v1")}

    def render(self, ctx, inputs, cfg: LogicSymmetryPlotCfg) -> list[PlotFigure]:
        from reader_workbench.domains.logic.logic_symmetry import render_logic_symmetry  # noqa: PLC0415

        figure = render_logic_symmetry(
            inputs["table"],
            title=cfg.title,
            dispersion=cfg.dispersion,
            encodings=cfg.encodings,
            ideals_overlay=cfg.ideals_overlay,
            visuals=cfg.visuals,
            figsize=cfg.figsize,
            dpi=cfg.dpi,
        )
        return [
            PlotFigure(
                fig=figure,
                filename=cfg.filename,
                ext=extension,
                dpi=cfg.dpi,
                description="Logic and asymmetry geometry over the configured four-state summary.",
            )
            for extension in cfg.format
        ]
