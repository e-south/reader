"""
SFXI vec8 heatmap plot plugin.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import Field

from reader_workbench.errors import SFXIError
from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


class SFXIVec8HeatmapCfg(PluginConfig):
    experiment_id: str | None = None
    title: str | None = None
    max_y_tick_labels: int = 80
    filename: str | None = None
    format: list[str] = Field(default_factory=lambda: ["pdf"])
    dpi: int = 300


class SFXIVec8HeatmapPlot(FigurePlotPlugin):
    ConfigModel = SFXIVec8HeatmapCfg

    @classmethod
    def input_ports(cls):
        return {"vec8": dataframe_input("vec8", "sfxi.vec8.v3")}

    def render(self, ctx, inputs, cfg: SFXIVec8HeatmapCfg) -> list[PlotFigure]:
        vec8: pd.DataFrame = inputs["vec8"]
        from reader_workbench.domains.logic.sfxi.vec8_heatmap import (  # noqa: PLC0415
            render_experiment_sfxi_vec8_heatmap,
        )

        experiment_id = cfg.experiment_id or _experiment_id_from_context(ctx)
        title = cfg.title or f"{experiment_id} SFXI vec8 heatmap"
        fig = render_experiment_sfxi_vec8_heatmap(
            vec8,
            experiment_id=experiment_id,
            title=title,
            max_y_tick_labels=cfg.max_y_tick_labels,
        )
        filename = cfg.filename or "sfxi_vec8_heatmap"
        return [PlotFigure(fig=fig, filename=filename, ext=ext, dpi=cfg.dpi) for ext in cfg.format]


def _experiment_id_from_context(ctx) -> str:
    exp_dir = getattr(ctx, "exp_dir", None)
    if isinstance(exp_dir, Path):
        return exp_dir.name
    if exp_dir is not None:
        name = Path(str(exp_dir)).name
        if name:
            return name
    raise SFXIError("plot/sfxi_vec8_heatmap requires experiment_id when no experiment directory is available.")
