from __future__ import annotations

import numpy as np
from pydantic import Field

from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin
from reader.workbench.ports import dataframe_input
from reader.workbench.registry import PluginConfig

_COLUMNS = ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")


class ResponseWindowSummaryCfg(PluginConfig):
    primary_reduction_id: str
    title: str = "Response-window summary"
    filename: str = "response_window_summary"
    format: list[str] = Field(default_factory=lambda: ["png"])
    dpi: int = 300


class ResponseWindowSummaryPlot(FigurePlotPlugin):
    ConfigModel = ResponseWindowSummaryCfg

    @classmethod
    def input_ports(cls):
        return {"designs": dataframe_input("designs", "plate_reader.response_window.designs.v3")}

    def render(self, ctx, inputs, cfg):
        from reader.plotting.style import use_style  # noqa: PLC0415

        frame = inputs["designs"]
        selected = frame.loc[
            frame["reduction_id"].astype(str).eq(cfg.primary_reduction_id) & ~frame["is_reference"].astype(bool)
        ].copy()
        if selected.empty:
            raise ValueError(
                f"response-window plot has no non-reference rows for reduction {cfg.primary_reduction_id!r}"
            )
        selected = selected.sort_values(["experiment_id", "design_id"], kind="stable")
        labels = (selected["experiment_id"].astype(str) + " :: " + selected["design_id"].astype(str)).tolist()
        values = selected.loc[:, _COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("response-window summary requires finite component values")
        with use_style({"axes_grid": False, "figure_figsize": (9.0, max(4.5, 0.32 * len(labels) + 2.0))}):
            import matplotlib.pyplot as plt  # noqa: PLC0415

            figure, axis = plt.subplots(constrained_layout=True)
            image = axis.imshow(values, aspect="auto", cmap="coolwarm")
            axis.set_xticks(range(len(_COLUMNS)), labels=_COLUMNS, rotation=45, ha="right")
            axis.set_yticks(range(len(labels)), labels=labels)
            axis.set_title(cfg.title)
            axis.set_xlabel("component")
            axis.set_ylabel("source :: design")
            figure.colorbar(image, ax=axis, label="summary value", shrink=0.8)
        return [
            PlotFigure(
                fig=figure,
                filename=cfg.filename,
                ext=extension,
                dpi=cfg.dpi,
                description="Primary event-relative response and anchored-magnitude components by source and design.",
            )
            for extension in cfg.format
        ]
