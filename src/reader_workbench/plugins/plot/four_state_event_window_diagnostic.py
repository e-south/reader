from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plotting.utils import slugify
from reader_workbench.plugins.plot._shared import FigurePlotPlugin
from reader_workbench.workbench.ports import dataframe_input
from reader_workbench.workbench.registry import PluginConfig


class FourStateEventWindowDiagnosticSubject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_experiment_id: str = Field(min_length=1)
    design_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _reject_blank_values(self) -> FourStateEventWindowDiagnosticSubject:
        required = {
            "source_experiment_id": self.source_experiment_id,
            "design_id": self.design_id,
            "filename": self.filename,
        }
        blank = [name for name, value in required.items() if not value.strip()]
        if self.title is not None and not self.title.strip():
            blank.append("title")
        if blank:
            raise ValueError(f"subject values must not be blank: {', '.join(blank)}")
        normalized_filename = slugify(self.filename)
        if not normalized_filename or not any(character.isalnum() for character in normalized_filename):
            raise ValueError("subject filename must contain a filesystem-safe character")
        return self


class FourStateEventWindowAxisLabels(BaseModel):
    """Protocol-derived labels for the three source trace panels."""

    model_config = ConfigDict(extra="forbid")

    growth: str = Field(min_length=1)
    response: str = Field(min_length=1)
    magnitude: str = Field(min_length=1)


class FourStateEventWindowDiagnosticCfg(PluginConfig):
    subjects: list[FourStateEventWindowDiagnosticSubject] = Field(min_length=1)
    primary_reduction_id: str = Field(min_length=1)
    pre_window_duration_h: float | None = Field(default=None, gt=0.0)
    axis_labels: FourStateEventWindowAxisLabels = Field(
        default_factory=lambda: FourStateEventWindowAxisLabels(
            growth="Signal",
            response="log$_2$(response signal)",
            magnitude="log$_2$(magnitude signal)",
        )
    )
    state_labels: dict[str, str] = Field(default_factory=lambda: {"00": "00", "10": "10", "01": "01", "11": "11"})
    reference_label: str = Field(default="reference", min_length=1)
    format: list[Literal["png", "pdf", "svg"]] = Field(default_factory=lambda: ["png"], min_length=1)
    dpi: int = Field(default=300, ge=1)

    @model_validator(mode="after")
    def _validate_subject_selection(self) -> FourStateEventWindowDiagnosticCfg:
        identities = [(subject.source_experiment_id, subject.design_id) for subject in self.subjects]
        if len(identities) != len(set(identities)):
            raise ValueError("subjects must contain unique source/design identities")
        filenames = [slugify(subject.filename).casefold() for subject in self.subjects]
        if len(filenames) != len(set(filenames)):
            raise ValueError("subjects must contain unique filenames")
        if set(self.state_labels) != {"00", "10", "01", "11"}:
            raise ValueError("state_labels must define exactly 00, 10, 01, and 11")
        if any(not isinstance(value, str) or not value.strip() for value in self.state_labels.values()):
            raise ValueError("state_labels values must be non-empty strings")
        return self


class FourStateEventWindowDiagnosticPlot(FigurePlotPlugin):
    ConfigModel = FourStateEventWindowDiagnosticCfg

    @classmethod
    def input_ports(cls):
        return {
            "traces": dataframe_input("traces", "plate_reader.four_state_event_window.traces.v3"),
            "designs": dataframe_input("designs", "plate_reader.four_state_event_window.designs.v4"),
        }

    def render(self, ctx, inputs, cfg: FourStateEventWindowDiagnosticCfg):
        from reader_workbench.domains.plate_reader.plots.four_state_event_window.diagnostic_render import (  # noqa: PLC0415
            render_four_state_event_window_diagnostic,
        )

        rendered: list[PlotFigure] = []
        for subject in cfg.subjects:
            figure = render_four_state_event_window_diagnostic(
                inputs["traces"],
                inputs["designs"],
                source_experiment_id=subject.source_experiment_id,
                design_id=subject.design_id,
                reduction_id=cfg.primary_reduction_id,
                pre_window_duration_h=cfg.pre_window_duration_h,
                axis_labels=cfg.axis_labels.model_dump(),
                state_labels=cfg.state_labels,
                reference_label=cfg.reference_label,
                title=subject.title,
            )
            rendered.extend(
                PlotFigure(
                    fig=figure,
                    filename=subject.filename,
                    ext=extension,
                    dpi=cfg.dpi,
                    description=(
                        "Event-relative growth, response, magnitude, and reduced components "
                        "for one source experiment and design."
                    ),
                )
                for extension in cfg.format
            )
        return rendered
