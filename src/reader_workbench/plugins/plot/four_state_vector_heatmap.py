"""
four-state vector heatmap plot plugin.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import Field

from reader_workbench.errors import FourStateVectorError
from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


class FourStateVectorHeatmapCfg(PluginConfig):
    experiment_id: str | None = None
    title: str | None = None
    max_y_tick_labels: int = 80
    filename: str | None = None
    format: list[str] = Field(default_factory=lambda: ["pdf"])
    dpi: int = 300


class FourStateVectorHeatmapPlot(FigurePlotPlugin):
    ConfigModel = FourStateVectorHeatmapCfg

    @classmethod
    def input_ports(cls):
        return {"vector": dataframe_input("vector", "logic.four_state_vector.v1")}

    def render(self, ctx, inputs, cfg: FourStateVectorHeatmapCfg) -> list[PlotFigure]:
        vector: pd.DataFrame = inputs["vector"]
        from reader_workbench.domains.logic.four_state_vector.heatmap import (  # noqa: PLC0415
            render_experiment_four_state_vector_heatmap,
        )

        experiment_id = cfg.experiment_id or _experiment_id_from_context(ctx)
        title = cfg.title or f"{experiment_id} four-state vector heatmap"
        fig = render_experiment_four_state_vector_heatmap(
            vector,
            experiment_id=experiment_id,
            title=title,
            max_y_tick_labels=cfg.max_y_tick_labels,
        )
        filename = cfg.filename or "four_state_vector_heatmap"
        return [PlotFigure(fig=fig, filename=filename, ext=ext, dpi=cfg.dpi) for ext in cfg.format]


def _experiment_id_from_context(ctx) -> str:
    exp_dir = getattr(ctx, "exp_dir", None)
    if isinstance(exp_dir, Path):
        return exp_dir.name
    if exp_dir is not None:
        name = Path(str(exp_dir)).name
        if name:
            return name
    raise FourStateVectorError(
        "plot/four_state_vector_heatmap requires experiment_id when no experiment directory is available."
    )
