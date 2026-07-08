"""
SFXI vec8 heatmap plot plugin.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import Field

from reader.domains.logic.sfxi.vec8_heatmap import render_experiment_sfxi_vec8_heatmap
from reader.errors import SFXIError
from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig


class SFXIVec8HeatmapCfg(PluginConfig):
    source_id: str | None = None
    title: str | None = None
    max_y_tick_labels: int = 80
    filename: str | None = None
    format: list[str] = Field(default_factory=lambda: ["pdf"])
    dpi: int = 300


class SFXIVec8HeatmapPlot(FigurePlotPlugin):
    ConfigModel = SFXIVec8HeatmapCfg

    @classmethod
    def input_ports(cls):
        return {"vec8": dataframe_input("vec8", "sfxi.vec8.v2")}

    def render(self, ctx, inputs, cfg: SFXIVec8HeatmapCfg) -> list[PlotFigure]:
        vec8: pd.DataFrame = inputs["vec8"]
        source_id = cfg.source_id or _source_id_from_context(ctx)
        title = cfg.title or f"{source_id} SFXI vec8 heatmap"
        fig = render_experiment_sfxi_vec8_heatmap(
            vec8,
            source_id=source_id,
            title=title,
            max_y_tick_labels=cfg.max_y_tick_labels,
        )
        filename = cfg.filename or "sfxi_vec8_heatmap"
        return [PlotFigure(fig=fig, filename=filename, ext=ext, dpi=cfg.dpi) for ext in cfg.format]


def _source_id_from_context(ctx) -> str:
    exp_dir = getattr(ctx, "exp_dir", None)
    if isinstance(exp_dir, Path):
        return exp_dir.name
    if exp_dir is not None:
        name = Path(str(exp_dir)).name
        if name:
            return name
    raise SFXIError("plot/sfxi_vec8_heatmap requires source_id when no experiment directory is available.")
