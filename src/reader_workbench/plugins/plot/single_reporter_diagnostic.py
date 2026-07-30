"""Four-panel diagnostic over a persisted single-reporter ratio record."""

from __future__ import annotations

import math
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from reader_workbench.domains.time_series import ObservationAggregationSpec, TemporalReductionSpec
from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plotting.utils import slugify
from reader_workbench.plugins.plot._shared import FigurePlotPlugin, PlotPartitionCfg, resolve_plot_partition_cfg
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig

_DESCRIPTION = (
    "Normalizer, reporter, and reporter-normalizer kinetics with an explicit endpoint or interval reduction, "
    "temporally reduced observation-unit values by condition, and visible normalizer QC."
)


class SingleReporterDiagnosticFigureCfg(BaseModel):
    figsize: tuple[float, float] = (14.0, 3.7)
    axis_label_size: float = Field(default=9.5, gt=0.0)
    tick_label_size: float = Field(default=8.5, gt=0.0)
    legend_fontsize: float = Field(default=8.5, gt=0.0)
    line_width: float = Field(default=1.7, gt=0.0)
    point_size: float = Field(default=28.0, gt=0.0)
    condition_tick_rotation: float = Field(default=20.0, ge=-90.0, le=90.0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_figure(self) -> SingleReporterDiagnosticFigureCfg:
        if len(self.figsize) != 2 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in self.figsize
        ):
            raise ValueError("single_reporter_diagnostic: fig.figsize must contain two positive finite values")
        return self


class SingleReporterDiagnosticCfg(PluginConfig):
    partition: PlotPartitionCfg = Field(default_factory=PlotPartitionCfg)
    condition_column: str = Field(default="treatment", min_length=1)
    condition_order: list[str] | None = None
    condition_order_ref: str | None = Field(default=None, min_length=1)
    temporal_reduction: dict[str, Any]
    observation_aggregation: dict[str, Any]
    time_column: str = Field(default="time", min_length=1)
    normalizer_channel: str = Field(min_length=1)
    reporter_channel: str = Field(min_length=1)
    ratio_channel: str = Field(min_length=1)
    filename_prefix: str = Field(default="single_reporter_diagnostic", min_length=1)
    format: list[Literal["png", "pdf", "svg"]] = Field(default_factory=lambda: ["png"], min_length=1)
    dpi: int = Field(default=300, ge=1)
    fig: SingleReporterDiagnosticFigureCfg = Field(default_factory=SingleReporterDiagnosticFigureCfg)

    @model_validator(mode="after")
    def validate_semantic_options(self) -> SingleReporterDiagnosticCfg:
        for field_name in (
            "condition_column",
            "time_column",
            "normalizer_channel",
            "reporter_channel",
            "ratio_channel",
            "filename_prefix",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"single_reporter_diagnostic: {field_name} must be a non-empty string")
            setattr(self, field_name, value)
        if len({self.normalizer_channel, self.reporter_channel, self.ratio_channel}) != 3:
            raise ValueError("single_reporter_diagnostic: normalizer, reporter, and ratio channels must be distinct")
        temporal = TemporalReductionSpec.from_mapping(self.temporal_reduction)
        aggregation = ObservationAggregationSpec.from_mapping(self.observation_aggregation)
        self.temporal_reduction = temporal.to_mapping()
        self.observation_aggregation = aggregation.to_mapping()
        if self.condition_order is not None and self.condition_order_ref is not None:
            raise ValueError(
                "single_reporter_diagnostic: condition_order and condition_order_ref are mutually exclusive"
            )
        if self.condition_order is not None:
            values = [str(value).strip() for value in self.condition_order]
            if not values or any(not value for value in values):
                raise ValueError("single_reporter_diagnostic: condition_order must contain non-empty labels")
            if len(set(values)) != len(values):
                raise ValueError("single_reporter_diagnostic: condition_order contains duplicate labels")
            self.condition_order = values
        if len(set(self.format)) != len(self.format):
            raise ValueError("single_reporter_diagnostic: format must not contain duplicates")
        return self


class SingleReporterDiagnosticPlot(FigurePlotPlugin):
    ConfigModel = SingleReporterDiagnosticCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "plate_reader.annotated.v1")}

    @classmethod
    def _resolve_condition_order(cls, *, experiment, cfg: SingleReporterDiagnosticCfg):
        return experiment.annotations.resolve_order_arg(
            order=cfg.condition_order,
            order_ref=cfg.condition_order_ref,
            column=cfg.condition_column,
            arg_name="condition_order",
        )

    @classmethod
    def validate_semantic_references(cls, *, experiment, cfg: SingleReporterDiagnosticCfg) -> None:
        experiment.annotations.resolve_plot_partition(partition=cfg.partition)
        cls._resolve_condition_order(experiment=experiment, cfg=cfg)

    def render(self, ctx, inputs, cfg: SingleReporterDiagnosticCfg) -> list[PlotFigure]:
        from reader_workbench.domains.plate_reader.plots.single_reporter_diagnostic import (  # noqa: PLC0415
            prepare_single_reporter_diagnostics,
        )
        from reader_workbench.domains.plate_reader.plots.single_reporter_diagnostic_render import (  # noqa: PLC0415
            render_single_reporter_diagnostic,
        )

        if ctx.experiment is None:
            raise ValueError("single_reporter_diagnostic requires experiment semantics in the run context")
        frame: pd.DataFrame = inputs["df"]
        partition = resolve_plot_partition_cfg(ctx=ctx, partition=cfg.partition)
        condition_order = type(self)._resolve_condition_order(experiment=ctx.experiment, cfg=cfg)
        unit_column, observation_column = _reduction_columns(ctx=ctx, frame=frame)
        diagnostics = prepare_single_reporter_diagnostics(
            frame,
            group_on=partition.group_by,
            collection_items=partition.collection_items,
            group_match=partition.match,
            condition_column=cfg.condition_column,
            condition_order=condition_order,
            unit_column=unit_column,
            observation_column=observation_column,
            time_column=cfg.time_column,
            normalizer_channel=cfg.normalizer_channel,
            reporter_channel=cfg.reporter_channel,
            ratio_channel=cfg.ratio_channel,
            temporal_reduction=TemporalReductionSpec.from_mapping(cfg.temporal_reduction),
            observation_aggregation=ObservationAggregationSpec.from_mapping(cfg.observation_aggregation),
        )
        artifact_names = _artifact_names(diagnostics, filename_prefix=cfg.filename_prefix)
        figures: list[PlotFigure] = []
        palette_book = getattr(ctx, "palette_book", None)
        for diagnostic in diagnostics:
            colors = palette_book.colors(len(diagnostic.condition_order)) if palette_book is not None else None
            figure = render_single_reporter_diagnostic(
                diagnostic,
                colors=colors,
                **cfg.fig.model_dump(),
            )
            figures.extend(
                PlotFigure(
                    fig=figure,
                    filename=artifact_names[diagnostic.group_label],
                    ext=extension,
                    dpi=cfg.dpi,
                    description=_DESCRIPTION,
                )
                for extension in cfg.format
            )
        return figures


def _reduction_columns(*, ctx, frame: pd.DataFrame) -> tuple[str, str]:
    if "position" not in frame.columns:
        raise ValueError("single_reporter_diagnostic: the annotated sample record requires position")
    evidence = getattr(ctx.experiment, "evidence", None)
    declared = getattr(evidence, "replicate_identity_field", None)
    if declared is not None:
        unit_column = str(declared).strip()
        if unit_column not in frame.columns:
            raise ValueError(
                "single_reporter_diagnostic: declared evidence.replicate_identity_field "
                f"{unit_column!r} is absent from the persisted ratio record"
            )
        return unit_column, "position"
    return "position", "position"


def _artifact_names(diagnostics, *, filename_prefix: str) -> dict[str, str]:
    names: dict[str, str] = {}
    owner_by_slug: dict[str, str] = {}
    for diagnostic in diagnostics:
        label = str(diagnostic.group_label)
        filename = filename_prefix if label == "all" else f"{filename_prefix}__{label}"
        normalized = slugify(filename)
        prior = owner_by_slug.get(normalized)
        if prior is not None:
            raise ValueError(
                "single_reporter_diagnostic: "
                f"partitions {prior!r} and {label!r} resolve to the same artifact name {normalized!r}"
            )
        names[label] = filename
        owner_by_slug[normalized] = label
    return names


__all__ = [
    "SingleReporterDiagnosticCfg",
    "SingleReporterDiagnosticFigureCfg",
    "SingleReporterDiagnosticPlot",
]
