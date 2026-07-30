from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field, model_validator

from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin, PlotPartitionCfg, resolve_plot_partition_cfg
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig


class TimeSeriesFigureCfg(PluginConfig):
    """Presentation options consumed by the time-series renderer."""

    figsize: tuple[float, float] | None = None
    dpi: int | None = Field(default=None, gt=0)
    ext: str | None = Field(default=None, min_length=1)
    rc: dict[str, Any] | None = None
    rasterize_zorder: float | None = None
    line_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_marker_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    observation_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    axis_label_size: float | None = Field(default=None, gt=0.0)
    title_fontsize: float | None = Field(default=None, gt=0.0)
    tick_label_size: float | None = Field(default=None, gt=0.0)
    legend_fontsize: float | None = Field(default=None, gt=0.0)
    legend_marker_size: float | None = Field(default=None, gt=0.0)
    mean_marker_size: float | None = Field(default=None, gt=0.0)
    observation_marker_size: float | None = Field(default=None, gt=0.0)
    line_width: float | None = Field(default=None, gt=0.0)
    top: float | None = Field(default=None, ge=0.0, le=1.0)
    bottom: float | None = Field(default=None, ge=0.0, le=1.0)
    left: float | None = Field(default=None, ge=0.0, le=1.0)
    right: float | None = Field(default=None, ge=0.0, le=1.0)
    wspace: float | None = Field(default=None, ge=0.0)
    hspace: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_geometry(self) -> TimeSeriesFigureCfg:
        if self.figsize is not None and any(value <= 0.0 for value in self.figsize):
            raise ValueError("time_series.fig.figsize values must be positive")
        if self.left is not None and self.right is not None and self.left >= self.right:
            raise ValueError("time_series.fig.left must be less than fig.right")
        if self.bottom is not None and self.top is not None and self.bottom >= self.top:
            raise ValueError("time_series.fig.bottom must be less than fig.top")
        return self


class TimeSeriesCfg(PluginConfig):
    x: str = "time"
    xlabel: str | None = None
    y: list[str] | None = None
    ylabel_map: dict[str, str] = Field(default_factory=dict)
    hue_label_map: dict[str, str] = Field(default_factory=dict)
    hue: str = "treatment"
    partition: PlotPartitionCfg = Field(default_factory=PlotPartitionCfg)
    fig: TimeSeriesFigureCfg = Field(default_factory=TimeSeriesFigureCfg)
    channels: list[str] | None = None
    add_sheet_line: bool = False
    sheet_line_kwargs: dict[str, Any] = Field(default_factory=dict)
    log_transform: bool | list[str] = False
    time_window: list[float] | None = None
    observation_interval_mass: float = Field(default=0.95, gt=0.0, lt=1.0)
    observation_interval_alpha: float = Field(default=0.15, ge=0.0, le=1.0)
    observation_resamples: int = Field(default=100, ge=1)
    observation_seed: int = 0
    legend_loc: str = "upper left"
    show_observations: bool = False
    shared_legend: bool = False
    filename: str | None = None


class TimeSeriesPlot(FigurePlotPlugin):
    ConfigModel = TimeSeriesCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    def render(self, ctx, inputs, cfg: TimeSeriesCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        from reader.domains.plate_reader.plots.time_series import plot_time_series  # noqa: PLC0415

        partition = resolve_plot_partition_cfg(ctx=ctx, partition=cfg.partition)

        return plot_time_series(
            df=df,
            x=cfg.x,
            xlabel=cfg.xlabel,
            y=cfg.y,
            ylabel_map=cfg.ylabel_map,
            hue_label_map=cfg.hue_label_map,
            hue=cfg.hue,
            channels=cfg.channels,
            group_on=partition.group_by,
            pool_sets=partition.collection_items,
            pool_match=partition.match,
            fig_kwargs=cfg.fig.model_dump(exclude_none=True),
            add_sheet_line=cfg.add_sheet_line,
            sheet_line_kwargs=cfg.sheet_line_kwargs,
            log_transform=cfg.log_transform,
            time_window=cfg.time_window,
            palette_book=ctx.palette_book,
            observation_interval_mass=cfg.observation_interval_mass,
            observation_interval_alpha=cfg.observation_interval_alpha,
            observation_resamples=cfg.observation_resamples,
            observation_seed=cfg.observation_seed,
            legend_loc=cfg.legend_loc,
            show_observations=cfg.show_observations,
            shared_legend=cfg.shared_legend,
            filename=cfg.filename,
        )
