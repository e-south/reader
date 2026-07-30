from __future__ import annotations

from importlib import import_module
from typing import Any

import pandas as pd
from pydantic import Field, model_validator

from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


def _load_snapshot_heatmap_impl():
    module = import_module("reader_workbench.domains.plate_reader.plots.snapshot_heatmap")
    return module.plot_snapshot_heatmap, module.prepare_snapshot_heatmap_inputs


class HeatmapCfg(PluginConfig):
    channel: str
    time: float
    x: str = "treatment"
    y: str = "design_id"
    order_x: list[str] | None = None
    order_y: list[str] | None = None
    order_x_ref: str | None = None
    order_y_ref: str | None = None
    square: bool = True
    vmin: float | None = None
    vmax: float | None = None
    value_transform: str | None = "none"  # "none" | "log2" | "log10"
    time_tolerance: float = 0.51
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None

    @model_validator(mode="after")
    def validate_order_sources(self) -> HeatmapCfg:
        if self.order_x is not None and self.order_x_ref is not None:
            raise ValueError("snapshot_heatmap: order_x and order_x_ref are mutually exclusive")
        if self.order_y is not None and self.order_y_ref is not None:
            raise ValueError("snapshot_heatmap: order_y and order_y_ref are mutually exclusive")
        return self


class SnapshotHeatmapPlot(FigurePlotPlugin):
    ConfigModel = HeatmapCfg

    @classmethod
    def input_ports(cls):
        return {
            "df": dataframe_input("df", "tidy.v1", optional=True),
            "fc": dataframe_input("fc", "fold_change.v1", optional=True),
        }

    def render(self, ctx, inputs, cfg: HeatmapCfg) -> list[PlotFigure]:
        if ctx.experiment is None:
            raise ValueError("snapshot_heatmap requires experiment semantics in the run context")
        df_in: pd.DataFrame | None = inputs.get("df")
        fc_in: pd.DataFrame | None = inputs.get("fc")
        plot_snapshot_heatmap, prepare_snapshot_heatmap_inputs = _load_snapshot_heatmap_impl()

        prepared = prepare_snapshot_heatmap_inputs(
            df_in=df_in,
            fc_in=fc_in,
            channel=cfg.channel,
            time=cfg.time,
            time_tolerance=cfg.time_tolerance,
            value_transform=cfg.value_transform,
            fig=cfg.fig,
            filename=cfg.filename,
            logger=ctx.logger,
        )
        channel = str(cfg.channel)
        df = prepared.frame
        filename = prepared.filename
        fig_kwargs = prepared.fig_kwargs
        resolved_order_x = ctx.experiment.annotations.resolve_order_arg(
            order=cfg.order_x,
            order_ref=cfg.order_x_ref,
            column=cfg.x,
            arg_name="order_x",
        )
        resolved_order_y = ctx.experiment.annotations.resolve_order_arg(
            order=cfg.order_y,
            order_ref=cfg.order_y_ref,
            column=cfg.y,
            arg_name="order_y",
        )
        return plot_snapshot_heatmap(
            df=df,
            channel=channel,
            time=cfg.time,
            x=cfg.x,
            y=cfg.y,
            order_x=resolved_order_x,
            order_y=resolved_order_y,
            square=cfg.square,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            fig_kwargs=fig_kwargs,
            filename=filename,
            logger=ctx.logger,
        )
