from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import Field

from reader.plotting.sinks import PlotFigure
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig

from ._shared import FigurePlotPlugin


class RetronSummaryCfg(PluginConfig):
    view: Literal["interaction", "heatmap", "stress_modulation", "pareto", "decomposition"]
    title: str = "Retron sponge summary"
    filename: str | None = None
    control_name: str = "tetO"
    no_stress_label: str = "H2O"
    relevant_only: bool = True
    metric: str | None = None
    state_order: list[str] | None = None
    burden_metric: Literal["D_growth_AUC", "T_growth_AUC", "T_finalOD"] = "D_growth_AUC"
    fig: dict[str, Any] = Field(default_factory=dict)


class RetronSummaryPlot(FigurePlotPlugin):
    ConfigModel = RetronSummaryCfg

    @classmethod
    def input_ports(cls):
        return {
            "summary": dataframe_input("summary", "plate_reader.sponge_summary.v1"),
            "trace": dataframe_input("trace", "plate_reader.sponge_trace.v1", optional=True),
        }

    def render(self, ctx, inputs, cfg: RetronSummaryCfg) -> list[PlotFigure]:
        summary: pd.DataFrame = inputs["summary"]
        trace: pd.DataFrame | None = inputs.get("trace")
        from reader.domains.plate_reader.plots.retron_sponge import plot_retron_sponge_summary  # noqa: PLC0415

        return plot_retron_sponge_summary(
            summary=summary,
            trace=trace,
            output_dir=None,
            view=cfg.view,
            title=cfg.title,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
            control_name=cfg.control_name,
            no_stress_label=cfg.no_stress_label,
            relevant_only=cfg.relevant_only,
            metric=cfg.metric,
            state_order=cfg.state_order,
            burden_metric=cfg.burden_metric,
            fig_kwargs=cfg.fig,
        )
