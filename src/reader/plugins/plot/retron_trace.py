from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from reader.plotting.sinks import PlotFigure
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig

from ._shared import FigurePlotPlugin


class RetronTraceCfg(PluginConfig):
    metrics: list[str] = Field(default_factory=list)
    title: str = "Retron sponge trace"
    filename: str | None = None
    control_name: str = "tetO"
    include_control: bool = False
    only_control: bool = False
    relevant_only: bool = False
    stress_order: list[str] | None = None
    fig: dict[str, Any] = Field(default_factory=dict)


class RetronTracePlot(FigurePlotPlugin):
    ConfigModel = RetronTraceCfg

    @classmethod
    def input_ports(cls):
        return {"trace": dataframe_input("trace", "plate_reader.sponge_trace.v1")}

    def render(self, ctx, inputs, cfg: RetronTraceCfg) -> list[PlotFigure]:
        trace: pd.DataFrame = inputs["trace"]
        from reader.domains.plate_reader.plots.retron_sponge import plot_retron_sponge_trace  # noqa: PLC0415

        return plot_retron_sponge_trace(
            trace=trace,
            output_dir=None,
            metrics=cfg.metrics,
            title=cfg.title,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
            control_name=cfg.control_name,
            include_control=cfg.include_control,
            only_control=cfg.only_control,
            relevant_only=cfg.relevant_only,
            stress_order=cfg.stress_order,
            fig_kwargs=cfg.fig,
        )
