from __future__ import annotations

from typing import Literal

from pydantic import Field

from reader_workbench.errors import ExecutionError
from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plotting.utils import slugify
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


class FourStateVectorDiagnosticCfg(PluginConfig):
    state_map_ref: str = Field(min_length=1)
    time_column: str = Field(default="time", min_length=1)
    growth_channel: str = Field(default="OD600", min_length=1)
    response_channel: str = Field(default="YFP/CFP", min_length=1)
    design_ids: list[str] | None = None
    trajectory_interval_mass: float = Field(default=0.95, gt=0.0, lt=1.0)
    trajectory_resamples: int = Field(default=300, ge=1)
    time_atol: float = Field(default=1e-9, ge=0.0)
    title: str | None = Field(default=None, min_length=1)
    filename: str = Field(default="four_state_vector_diagnostic", min_length=1)
    format: list[Literal["png", "pdf", "svg"]] = Field(default_factory=lambda: ["png"], min_length=1)
    dpi: int = Field(default=300, ge=1)
    figsize: tuple[float, float] = (15.0, 5.5)


class FourStateVectorDiagnosticPlot(FigurePlotPlugin):
    """Render a record-driven diagnostic without recomputing vector."""

    ConfigModel = FourStateVectorDiagnosticCfg

    @classmethod
    def input_ports(cls):
        return {
            "df": dataframe_input("df", "plate_reader.annotated.v1"),
            "vector": dataframe_input("vector", "logic.four_state_vector.v1"),
        }

    def render(self, ctx, inputs, cfg: FourStateVectorDiagnosticCfg) -> list[PlotFigure]:
        if ctx is None or getattr(ctx, "experiment", None) is None:
            raise ExecutionError("plot/four_state_vector_diagnostic requires experiment annotation semantics")
        state_space = ctx.experiment.annotations.resolve_ordered_state_space(ref=cfg.state_map_ref)
        if tuple(state_space.state_ids) != ("00", "10", "01", "11"):
            raise ExecutionError("plot/four_state_vector_diagnostic requires ordered states 00, 10, 01, 11")

        from reader_workbench.domains.logic.four_state_vector.diagnostic import (  # noqa: PLC0415
            prepare_four_state_vector_diagnostics,
            render_four_state_vector_diagnostic,
        )

        prepared = prepare_four_state_vector_diagnostics(
            inputs["df"],
            inputs["vector"],
            treatment_column=state_space.column,
            treatment_map=state_space.source_values,
            treatment_case_sensitive=state_space.case_sensitive,
            time_column=cfg.time_column,
            growth_channel=cfg.growth_channel,
            response_channel=cfg.response_channel,
            design_ids=cfg.design_ids,
            time_atol=cfg.time_atol,
            trajectory_interval_mass=cfg.trajectory_interval_mass,
            trajectory_resamples=cfg.trajectory_resamples,
        )
        filenames = [_design_filename(cfg.filename, item.design_id) for item in prepared]
        if len(set(filenames)) != len(filenames):
            raise ExecutionError("plot/four_state_vector_diagnostic design ids resolve to duplicate artifact filenames")

        figures: list[PlotFigure] = []
        for item, filename in zip(prepared, filenames, strict=True):
            figure = render_four_state_vector_diagnostic(
                item,
                title=cfg.title,
                figsize=cfg.figsize,
                dpi=cfg.dpi,
            )
            description = (
                f"Growth and response trajectories with persisted vector components for design {item.design_id} "
                f"at {item.selected_time_h:g} hours."
            )
            figures.extend(
                PlotFigure(
                    fig=figure,
                    filename=filename,
                    ext=extension,
                    dpi=cfg.dpi,
                    description=description,
                )
                for extension in cfg.format
            )
        return figures


def _design_filename(base: str, design_id: str) -> str:
    base_slug = slugify(base)
    if not base_slug:
        raise ExecutionError(f"plot/four_state_vector_diagnostic cannot derive a filename from base {base!r}")
    design_slug = slugify(design_id)
    if not design_slug:
        raise ExecutionError(f"plot/four_state_vector_diagnostic cannot derive a filename from design id {design_id!r}")
    return f"{base_slug}--{design_slug}"
