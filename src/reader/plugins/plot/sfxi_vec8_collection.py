from __future__ import annotations

from pydantic import Field

from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig


class SFXIVec8CollectionHeatmapCfg(PluginConfig):
    title: str = "SFXI vec8 collection"
    max_y_tick_labels: int = 80
    filename: str = "sfxi_vec8_collection"
    format: list[str] = Field(default_factory=lambda: ["png"])
    dpi: int = 300


class SFXIVec8CollectionHeatmapPlot(FigurePlotPlugin):
    ConfigModel = SFXIVec8CollectionHeatmapCfg

    @classmethod
    def input_ports(cls):
        return {"vec8": dataframe_input("vec8", "sfxi.vec8_collection.v2")}

    def render(self, ctx, inputs, cfg):
        from reader.domains.logic.sfxi.vec8_aggregate import render_sfxi_vec8_heatmap  # noqa: PLC0415

        figure = render_sfxi_vec8_heatmap(
            inputs["vec8"],
            title=cfg.title,
            max_y_tick_labels=cfg.max_y_tick_labels,
        )
        return [
            PlotFigure(
                fig=figure,
                filename=cfg.filename,
                ext=extension,
                dpi=cfg.dpi,
                description="Cross-experiment SFXI vec8 heatmap.",
            )
            for extension in cfg.format
        ]
