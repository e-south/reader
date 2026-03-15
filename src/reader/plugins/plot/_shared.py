"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/_shared.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel

from reader.core.plot_sinks import PlotFigure, normalize_plot_figures, save_plot_figures
from reader.core.registry import Plugin
from reader.core.semantics import resolve_plot_partition


class PlotPartitionCfg(BaseModel):
    by: str | None = None
    collection_ref: str | None = None
    match: Literal["exact", "contains", "startswith", "endswith", "regex"] = "exact"

    model_config = {"extra": "forbid"}


class FigurePlotPlugin(Plugin):
    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    @abstractmethod
    def render(self, ctx, inputs, cfg) -> list[PlotFigure]:
        raise NotImplementedError

    def run(self, ctx, inputs, cfg):
        return save_rendered_figures(ctx=ctx, figures=self.render(ctx, inputs, cfg), plot_key=self.key)


def resolve_plot_partition_cfg(*, ctx, partition: PlotPartitionCfg):
    return resolve_plot_partition(partition=partition, assay=dict(ctx.assay or {}))


def save_rendered_figures(*, ctx, figures: list[PlotFigure], plot_key: str) -> dict[str, list[str] | None]:
    normalized = normalize_plot_figures(figures, where=f"plot/{plot_key}")
    saved = save_plot_figures(normalized, ctx.plots_dir)
    return {"files": [str(path) for path in saved] if saved else None}
