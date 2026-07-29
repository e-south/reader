"""Two-panel plot (time series + snapshot barplot)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import Field, model_validator

from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin, PlotPartitionCfg, resolve_plot_partition_cfg
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig


class TSAndSnapFigureCfg(PluginConfig):
    """Figure and presentation options accepted by the composite renderer."""

    figsize: tuple[float, float] | None = None
    dpi: float | None = Field(default=None, gt=0)
    facecolor: str | None = None
    edgecolor: str | None = None
    frameon: bool | None = None
    clear: bool | None = None
    constrained_layout: bool | None = None
    layout: str | None = None
    rc: dict[str, Any] | None = None
    ext: str | None = None
    axis_label_size: float | None = Field(default=None, gt=0)
    tick_label_size: float | None = Field(default=None, gt=0)
    legend_fontsize: float | None = Field(default=None, gt=0)
    legend_marker_size: float | None = Field(default=None, gt=0)
    line_width: float | None = Field(default=None, gt=0)
    mean_marker_size: float | None = Field(default=None, gt=0)
    mean_marker_every: int | None = Field(default=None, ge=1)
    replicate_marker_size: float | None = Field(default=None, gt=0)
    style_legend_loc: str | None = None
    style_legend_title: str | None = None
    snap_tick_rotation: float | None = None
    ts_title: str | None = None
    snap_title: str | None = None
    line_alpha: float | None = Field(default=None, ge=0, le=1)
    mean_marker_alpha: float | None = Field(default=None, ge=0, le=1)
    replicate_alpha: float | None = Field(default=None, ge=0, le=1)
    suptitle_y: float | None = None


class TSAndSnapCfg(PluginConfig):
    # grouping
    partition: PlotPartitionCfg = Field(default_factory=PlotPartitionCfg)
    group_layout: Literal["separate", "paired_row"] = "separate"

    # time series (left)
    ts_x: str = "time"
    ts_channel: str
    ts_hue: str
    ts_style: str | None = None
    order_hue: list[str] | None = None
    order_hue_ref: str | None = None
    order_style: list[str] | None = None
    order_style_ref: str | None = None
    ts_time_window: list[float] | None = None
    ts_add_sheet_line: bool = False
    ts_sheet_line_kwargs: dict[str, Any] = Field(default_factory=dict)
    ts_mark_snap_time: bool = False
    ts_snap_line_kwargs: dict[str, Any] = Field(default_factory=dict)
    ts_log_transform: bool | list[str] = False
    ts_ci: float = 95.0
    ts_ci_alpha: float = 0.15
    ts_ci_boot: int = Field(default=100, ge=1)
    ts_ci_seed: int = 0
    ts_show_replicates: bool = False
    ts_legend_loc: str = "upper right"

    # snapshot (right)
    snap_x: str = "treatment"
    snap_channel: str | None = None
    snap_hue: str | None = None
    order_x: list[str] | None = None
    order_x_ref: str | None = None
    order_snap_hue: list[str] | None = None
    order_snap_hue_ref: str | None = None
    snap_time: float = Field(..., description="Snapshot time (hours); required for deterministic plotting.")
    snap_agg: Literal["mean", "median"] = "mean"
    snap_err: Literal["sem", "iqr", "none"] = "sem"
    snap_time_tolerance: float = 0.51
    snap_show_legend: bool = False
    snap_legend_loc: str = "upper right"
    snap_color_by_x: bool = False
    square_panels: bool = False

    # figure
    fig: TSAndSnapFigureCfg = Field(default_factory=TSAndSnapFigureCfg)
    filename: str | None = None
    title: str | None = None

    @model_validator(mode="after")
    def validate_order_sources(self) -> TSAndSnapCfg:
        for arg_name in ("order_hue", "order_style", "order_x", "order_snap_hue"):
            if getattr(self, arg_name) is not None and getattr(self, f"{arg_name}_ref") is not None:
                raise ValueError(f"ts_and_snap: {arg_name} and {arg_name}_ref are mutually exclusive")
        if self.order_style is not None and self.ts_style is None:
            raise ValueError("ts_and_snap: order_style requires ts_style")
        if self.order_style_ref is not None and self.ts_style is None:
            raise ValueError("ts_and_snap: order_style_ref requires ts_style")
        if self.order_snap_hue is not None and self.snap_hue is None:
            raise ValueError("ts_and_snap: order_snap_hue requires snap_hue")
        if self.order_snap_hue_ref is not None and self.snap_hue is None:
            raise ValueError("ts_and_snap: order_snap_hue_ref requires snap_hue")
        if self.snap_color_by_x and self.snap_hue is not None:
            raise ValueError("ts_and_snap: snap_color_by_x requires snap_hue to be omitted")
        return self


class TSAndSnapPlot(FigurePlotPlugin):
    ConfigModel = TSAndSnapCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def _resolve_semantic_orders(cls, *, experiment, cfg: TSAndSnapCfg):
        return (
            experiment.annotations.resolve_order_arg(
                order=cfg.order_hue,
                order_ref=cfg.order_hue_ref,
                column=cfg.ts_hue,
                arg_name="order_hue",
            ),
            experiment.annotations.resolve_order_arg(
                order=cfg.order_style,
                order_ref=cfg.order_style_ref,
                column=cfg.ts_style,
                arg_name="order_style",
            ),
            experiment.annotations.resolve_order_arg(
                order=cfg.order_x,
                order_ref=cfg.order_x_ref,
                column=cfg.snap_x,
                arg_name="order_x",
            ),
            experiment.annotations.resolve_order_arg(
                order=cfg.order_snap_hue,
                order_ref=cfg.order_snap_hue_ref,
                column=cfg.snap_hue,
                arg_name="order_snap_hue",
            ),
        )

    @classmethod
    def validate_semantic_references(cls, *, experiment, cfg: TSAndSnapCfg) -> None:
        cls._resolve_semantic_orders(experiment=experiment, cfg=cfg)

    def render(self, ctx, inputs, cfg: TSAndSnapCfg) -> list[PlotFigure]:
        df: pd.DataFrame = inputs["df"]
        from reader.domains.plate_reader.plots.ts_and_snap import plot_ts_and_snap  # noqa: PLC0415

        partition = resolve_plot_partition_cfg(ctx=ctx, partition=cfg.partition)
        if ctx.experiment is None:
            raise ValueError("ts_and_snap requires experiment semantics in the run context")
        resolved_order_hue, resolved_order_style, resolved_order_x, resolved_order_snap_hue = type(
            self
        )._resolve_semantic_orders(experiment=ctx.experiment, cfg=cfg)

        return plot_ts_and_snap(
            df=df,
            group_on=partition.group_by,
            pool_sets=partition.collection_items,
            pool_match=partition.match,
            group_layout=cfg.group_layout,
            # ts (left)
            ts_x=cfg.ts_x,
            ts_channel=cfg.ts_channel,
            ts_hue=cfg.ts_hue,
            ts_style=cfg.ts_style,
            order_hue=resolved_order_hue,
            order_style=resolved_order_style,
            ts_time_window=cfg.ts_time_window,
            ts_add_sheet_line=cfg.ts_add_sheet_line,
            ts_sheet_line_kwargs=cfg.ts_sheet_line_kwargs,
            ts_mark_snap_time=cfg.ts_mark_snap_time,
            ts_snap_line_kwargs=cfg.ts_snap_line_kwargs,
            ts_log_transform=cfg.ts_log_transform,
            ts_ci=cfg.ts_ci,
            ts_ci_alpha=cfg.ts_ci_alpha,
            ts_ci_boot=cfg.ts_ci_boot,
            ts_ci_seed=cfg.ts_ci_seed,
            ts_show_replicates=cfg.ts_show_replicates,
            ts_legend_loc=cfg.ts_legend_loc,
            # snap (right)
            snap_x=cfg.snap_x,
            snap_channel=cfg.snap_channel,
            snap_hue=cfg.snap_hue,
            order_x=resolved_order_x,
            order_snap_hue=resolved_order_snap_hue,
            snap_time=cfg.snap_time,
            snap_agg=cfg.snap_agg,
            snap_err=cfg.snap_err,
            snap_time_tolerance=cfg.snap_time_tolerance,
            snap_show_legend=cfg.snap_show_legend,
            snap_legend_loc=cfg.snap_legend_loc,
            snap_color_by_x=cfg.snap_color_by_x,
            square_panels=cfg.square_panels,
            # fig
            fig_kwargs=cfg.fig.model_dump(exclude_none=True),
            filename=cfg.filename,
            title=cfg.title,
            palette_book=ctx.palette_book,
        )
