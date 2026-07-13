"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/_shared.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

from pydantic import BaseModel

from reader.plotting.sinks import PlotFigure, normalize_plot_figures, save_plot_figures
from reader.workbench.ports import file_bundle_output
from reader.workbench.records import PathDescription
from reader.workbench.registry import Plugin


class PlotPartitionCfg(BaseModel):
    by: str | None = None
    collection_ref: str | None = None
    match: Literal["exact", "contains", "startswith", "endswith", "regex"] = "exact"

    model_config = {"extra": "forbid"}


class FigurePlotPlugin(Plugin):
    @classmethod
    def output_ports(cls):
        return {"artifacts": file_bundle_output("artifacts")}

    @abstractmethod
    def render(self, ctx, inputs, cfg) -> list[PlotFigure]:
        raise NotImplementedError

    def run(self, ctx, inputs, cfg):
        return save_rendered_figures(ctx=ctx, figures=self.render(ctx, inputs, cfg), plot_key=self.plugin_key)


def resolve_plot_partition_cfg(*, ctx, partition: PlotPartitionCfg):
    if ctx.experiment is None:
        raise ValueError("plot partition resolution requires experiment semantics in the run context")
    return ctx.experiment.annotations.resolve_plot_partition(partition=partition)


def save_rendered_figures(*, ctx, figures: list[PlotFigure], plot_key: str) -> dict[str, list[str | PathDescription]]:
    normalized = normalize_plot_figures(figures, where=f"plot/{plot_key}")
    saved = save_plot_figures(normalized, ctx.plots_dir)
    artifacts: list[str | PathDescription] = []
    for figure, path in zip(normalized, saved, strict=True):
        if figure.description is None:
            artifacts.append(str(path))
        else:
            artifacts.append(PathDescription(path=path, description=figure.description))
    return {"artifacts": artifacts}
