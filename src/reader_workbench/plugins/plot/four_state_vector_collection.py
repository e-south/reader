from __future__ import annotations

from pydantic import Field

from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


class FourStateVectorCollectionHeatmapCfg(PluginConfig):
    title: str = "Four-state vector collection"
    max_y_tick_labels: int = 80
    filename: str = "four_state_vector_collection"
    format: list[str] = Field(default_factory=lambda: ["png"])
    dpi: int = 300


class FourStateVectorCollectionHeatmapPlot(FigurePlotPlugin):
    ConfigModel = FourStateVectorCollectionHeatmapCfg

    @classmethod
    def input_ports(cls):
        return {"vectors": dataframe_input("vectors", "logic.four_state_vector_collection.v1")}

    def render(self, ctx, inputs, cfg):
        from reader_workbench.domains.logic.four_state_vector.collection import (  # noqa: PLC0415
            render_four_state_vector_collection_heatmap,
        )

        figure = render_four_state_vector_collection_heatmap(
            inputs["vectors"],
            title=cfg.title,
            max_y_tick_labels=cfg.max_y_tick_labels,
        )
        return [
            PlotFigure(
                fig=figure,
                filename=cfg.filename,
                ext=extension,
                dpi=cfg.dpi,
                description="Cross-experiment four-state vector heatmap.",
            )
            for extension in cfg.format
        ]
