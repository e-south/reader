"""Persisted dual-reporter kinetics and endpoint triptych."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from reader.plotting.sinks import PlotFigure
from reader.plotting.utils import slugify
from reader.plugins.plot._shared import FigurePlotPlugin
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig

_DESCRIPTION = (
    "Growth and reporter-ratio kinetics with bootstrap confidence intervals, "
    "plus observed endpoint values and mean with sample standard deviation."
)


class DualReporterTriptychCfg(PluginConfig):
    design_column: str = Field(default="design_id", min_length=1)
    treatment_column: str = Field(default="treatment", min_length=1)
    time_column: str = Field(default="time", min_length=1)
    growth_channel: str = Field(default="OD600", min_length=1)
    ratio_channel: str = Field(default="YFP/CFP", min_length=1)
    snapshot_channel: str | None = Field(default=None, min_length=1)
    snapshot_time_h: float
    snapshot_time_mode: Literal["nearest", "last_before", "first_after", "exact"] = "nearest"
    snapshot_time_tolerance_h: float | None = Field(default=0.51, ge=0.0)
    treatment_order: list[str] | None = None
    treatment_order_ref: str | None = Field(default=None, min_length=1)
    trajectory_ci: float = Field(default=95.0, gt=0.0, lt=100.0)
    trajectory_bootstraps: int = Field(default=300, ge=1)
    filename_prefix: str = Field(default="dual_reporter_triptych", min_length=1)
    format: list[Literal["png", "pdf", "svg"]] = Field(default_factory=lambda: ["png"], min_length=1)
    dpi: int = Field(default=300, ge=1)
    figsize: tuple[float, float] = (10.5, 3.5)

    @model_validator(mode="after")
    def validate_semantic_options(self) -> DualReporterTriptychCfg:
        filename_prefix = self.filename_prefix.strip()
        if not filename_prefix:
            raise ValueError("dual_reporter_triptych: filename_prefix must be a non-empty string")
        self.filename_prefix = filename_prefix
        if self.treatment_order is not None and self.treatment_order_ref is not None:
            raise ValueError("dual_reporter_triptych: treatment_order and treatment_order_ref are mutually exclusive")
        if self.treatment_order is not None:
            normalized = [str(value).strip() for value in self.treatment_order]
            if not normalized or any(not value for value in normalized):
                raise ValueError("dual_reporter_triptych: treatment_order must contain non-empty strings")
            if len(set(normalized)) != len(normalized):
                raise ValueError("dual_reporter_triptych: treatment_order contains duplicate labels")
            self.treatment_order = normalized
        if any(not value > 0.0 for value in self.figsize):
            raise ValueError("dual_reporter_triptych: figsize values must be positive")
        if len(set(self.format)) != len(self.format):
            raise ValueError("dual_reporter_triptych: format must not contain duplicates")
        return self


class DualReporterTriptychPlot(FigurePlotPlugin):
    ConfigModel = DualReporterTriptychCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def validate_semantic_references(cls, *, experiment, cfg: DualReporterTriptychCfg) -> None:
        if cfg.treatment_order_ref is not None:
            experiment.annotations.resolve_order_arg(
                order=None,
                order_ref=cfg.treatment_order_ref,
                column=cfg.treatment_column,
                arg_name="treatment_order",
            )

    def render(self, ctx, inputs, cfg: DualReporterTriptychCfg) -> list[PlotFigure]:
        from reader.domains.plate_reader.analysis.timepoints import (  # noqa: PLC0415
            infer_acquisition_transition_time_h,
        )
        from reader.domains.plate_reader.ordering import order_levels  # noqa: PLC0415
        from reader.domains.plate_reader.plots.dual_reporter_triptych import (  # noqa: PLC0415
            build_triptych_data,
            choose_time,
        )
        from reader.domains.plate_reader.plots.dual_reporter_triptych_render import (  # noqa: PLC0415
            render_dual_reporter_triptych,
        )

        frame: pd.DataFrame = inputs["df"]
        _require_columns(
            frame,
            (
                cfg.design_column,
                cfg.treatment_column,
                cfg.time_column,
                "channel",
                "value",
                "position",
            ),
        )
        design_values = order_levels(_nonempty_strings(frame[cfg.design_column]))
        if not design_values:
            raise ValueError(f"dual_reporter_triptych: no non-empty values in {cfg.design_column!r}")
        design_series = frame[cfg.design_column].astype("string").str.strip()
        filename_by_design = _artifact_names_by_design(
            design_values,
            filename_prefix=cfg.filename_prefix,
        )
        treatment_order = _resolve_treatment_order(ctx=ctx, cfg=cfg)

        figures: list[PlotFigure] = []
        for design_value in design_values:
            selected = frame[design_series == design_value].copy()
            time_values = pd.to_numeric(selected[cfg.time_column], errors="coerce").dropna().unique().tolist()
            selected_time = choose_time(time_values, cfg.snapshot_time_h, cfg.snapshot_time_mode)
            if selected_time is None:
                raise ValueError(
                    "dual_reporter_triptych: "
                    f"design {design_value!r} has no time matching {cfg.snapshot_time_h:g} h "
                    f"with mode {cfg.snapshot_time_mode!r}"
                )
            delta = abs(float(selected_time) - float(cfg.snapshot_time_h))
            if cfg.snapshot_time_tolerance_h is not None and delta > cfg.snapshot_time_tolerance_h:
                raise ValueError(
                    "dual_reporter_triptych: "
                    f"design {design_value!r} selected {selected_time:g} h for target "
                    f"{cfg.snapshot_time_h:g} h, outside snapshot_time_tolerance_h="
                    f"{cfg.snapshot_time_tolerance_h:g}"
                )

            data = build_triptych_data(
                selected,
                time_col=cfg.time_column,
                treatment_col=cfg.treatment_column,
                growth_channel=cfg.growth_channel,
                ratio_channel=cfg.ratio_channel,
                snapshot_channel=cfg.snapshot_channel,
                snapshot_time=float(selected_time),
                treatment_order=treatment_order,
                trajectory_ci=cfg.trajectory_ci,
                trajectory_bootstraps=cfg.trajectory_bootstraps,
            )
            palette_book = getattr(ctx, "palette_book", None)
            colors = palette_book.colors(len(data.treatment_order)) if palette_book is not None else None
            figure = render_dual_reporter_triptych(
                data,
                time_col=cfg.time_column,
                treatment_col=cfg.treatment_column,
                acquisition_transition_time_h=infer_acquisition_transition_time_h(
                    selected,
                    time_col=cfg.time_column,
                ),
                title=design_value,
                colors=colors,
                figsize=cfg.figsize,
            )
            filename = filename_by_design[design_value]
            figures.extend(
                PlotFigure(
                    fig=figure,
                    filename=filename,
                    ext=extension,
                    dpi=cfg.dpi,
                    description=_DESCRIPTION,
                )
                for extension in cfg.format
            )
        return figures


def _resolve_treatment_order(*, ctx, cfg: DualReporterTriptychCfg) -> list[str] | None:
    if cfg.treatment_order_ref is None:
        return cfg.treatment_order
    experiment = getattr(ctx, "experiment", None)
    if experiment is None:
        raise ValueError("dual_reporter_triptych: treatment_order_ref requires experiment semantics")
    return experiment.annotations.resolve_order_arg(
        order=None,
        order_ref=cfg.treatment_order_ref,
        column=cfg.treatment_column,
        arg_name="treatment_order",
    )


def _nonempty_strings(series: pd.Series) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.dropna().tolist():
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none"} or text in seen:
            continue
        values.append(text)
        seen.add(text)
    return values


def _artifact_names_by_design(design_values: list[str], *, filename_prefix: str) -> dict[str, str]:
    by_design: dict[str, str] = {}
    owner_by_name: dict[str, str] = {}
    for design_value in design_values:
        name = f"{filename_prefix}__{design_value}"
        normalized_name = slugify(name)
        prior = owner_by_name.get(normalized_name)
        if prior is not None:
            raise ValueError(
                "dual_reporter_triptych: "
                f"designs {prior!r} and {design_value!r} resolve to the same artifact name {normalized_name!r}"
            )
        by_design[design_value] = name
        owner_by_name[normalized_name] = design_value
    return by_design


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in dict.fromkeys(columns) if column not in frame.columns]
    if missing:
        raise ValueError("dual_reporter_triptych: missing column(s): " + ", ".join(missing))


__all__ = ["DualReporterTriptychCfg", "DualReporterTriptychPlot"]
