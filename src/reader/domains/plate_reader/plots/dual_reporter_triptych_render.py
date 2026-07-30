"""Static Matplotlib rendering for the dual-reporter triptych contract."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from reader.plotting.style import use_style

from .dual_reporter_triptych import DualReporterTriptychData

_DEFAULT_COLORS = ("#334155", "#0f766e", "#2563eb", "#be123c")
_LINESTYLES = ("-", "--", "-.", ":")


def render_dual_reporter_triptych(
    data: DualReporterTriptychData,
    *,
    time_col: str,
    treatment_col: str,
    acquisition_transition_time_h: float | None = None,
    title: str | None = None,
    colors: Sequence[str] | None = None,
    figsize: tuple[float, float] = (10.5, 3.5),
) -> Figure:
    """Render two kinetic panels and one observed-value snapshot panel."""

    if data.od600_time.empty:
        raise ValueError("dual_reporter_triptych: no growth time-series data available")
    if data.ratio_time.empty:
        raise ValueError("dual_reporter_triptych: no ratio time-series data available")
    if data.snapshot_stats.empty:
        raise ValueError("dual_reporter_triptych: no snapshot data available")

    order = list(data.treatment_order)
    palette = list(colors or _DEFAULT_COLORS)
    if not palette:
        raise ValueError("dual_reporter_triptych: colors must contain at least one color")
    color_map = {value: palette[index % len(palette)] for index, value in enumerate(order)}

    with use_style(
        rc={
            "axes_titleweight": "regular",
            "axes_titlesize": 11.0,
            "axes_labelsize": 10.0,
            "font_size": 10.0,
            "legend_fontsize": 9.0,
            "legend_title_fontsize": 9.0,
            "xtick_labelsize": 9.0,
            "ytick_labelsize": 9.0,
        },
        color_cycle=palette,
    ):
        figure, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        _draw_time_panel(
            axes[0],
            data.od600_time,
            time_col=time_col,
            treatment_col=treatment_col,
            order=order,
            color_map=color_map,
            snapshot_time=data.snapshot_time,
            acquisition_transition_time_h=acquisition_transition_time_h,
            title=f"{data.growth_channel} kinetics",
            ylabel=data.growth_channel,
        )
        _draw_time_panel(
            axes[1],
            data.ratio_time,
            time_col=time_col,
            treatment_col=treatment_col,
            order=order,
            color_map=color_map,
            snapshot_time=data.snapshot_time,
            acquisition_transition_time_h=acquisition_transition_time_h,
            title=f"{data.ratio_channel} kinetics",
            ylabel=data.ratio_channel,
        )
        _draw_snapshot_panel(
            axes[2],
            data,
            treatment_col=treatment_col,
            color_map=color_map,
            title=f"{data.snapshot_channel} snapshot at {data.snapshot_time:g} h",
        )

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="outside lower center",
                ncols=max(1, min(len(handles), 4)),
                title=str(treatment_col).replace("_", " ").capitalize(),
            )
        if title:
            figure.suptitle(str(title), fontweight="normal")
        return figure


def _draw_time_panel(
    axis,
    frame,
    *,
    time_col: str,
    treatment_col: str,
    order: list[str],
    color_map: dict[str, str],
    snapshot_time: float,
    acquisition_transition_time_h: float | None,
    title: str,
    ylabel: str,
) -> None:
    for index, treatment in enumerate(order):
        subset = frame[frame[treatment_col].astype(str) == str(treatment)].sort_values(time_col)
        if subset.empty:
            continue
        x = subset[time_col].to_numpy(dtype=float)
        mean = subset["y_mean"].to_numpy(dtype=float)
        lower = subset["y_lo"].to_numpy(dtype=float)
        upper = subset["y_hi"].to_numpy(dtype=float)
        color = color_map[treatment]
        axis.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)
        axis.plot(
            x,
            mean,
            color=color,
            linestyle=_LINESTYLES[index % len(_LINESTYLES)],
            linewidth=1.7,
            label=treatment,
        )

    axis.axvline(float(snapshot_time), color="#111827", linewidth=1.0)
    if acquisition_transition_time_h is not None:
        transition = float(acquisition_transition_time_h)
        if np.isfinite(transition):
            axis.axvline(transition, color="#64748b", linewidth=1.0, linestyle=(0, (5, 3)))
    axis.set_title(title)
    axis.set_xlabel("Time (h)")
    axis.set_ylabel(ylabel)


def _draw_snapshot_panel(
    axis,
    data: DualReporterTriptychData,
    *,
    treatment_col: str,
    color_map: dict[str, str],
    title: str,
) -> None:
    order = list(data.treatment_order)
    x_positions = {value: float(index) for index, value in enumerate(order)}

    for treatment in order:
        subset = data.snapshot_points[data.snapshot_points[treatment_col].astype(str) == str(treatment)].sort_values(
            "replicate_index"
        )
        offsets = np.linspace(-0.12, 0.12, len(subset)) if len(subset) > 1 else np.zeros(len(subset))
        if not subset.empty:
            axis.scatter(
                x_positions[treatment] + offsets,
                subset["value"].to_numpy(dtype=float),
                s=28,
                facecolors="white",
                edgecolors=color_map[treatment],
                linewidths=1.0,
                zorder=3,
            )

    for _, row in data.snapshot_stats.iterrows():
        treatment = str(row[treatment_col])
        if treatment not in x_positions:
            continue
        center = x_positions[treatment]
        mean = float(row["y_mean"])
        lower = float(row["y_lo"])
        upper = float(row["y_hi"])
        if not all(np.isfinite(value) for value in (mean, lower, upper)):
            continue
        color = color_map[treatment]
        axis.plot([center, center], [lower, upper], color=color, linewidth=1.0, zorder=4)
        axis.plot([center - 0.05, center + 0.05], [lower, lower], color=color, linewidth=1.0, zorder=4)
        axis.plot([center - 0.05, center + 0.05], [upper, upper], color=color, linewidth=1.0, zorder=4)
        axis.plot([center - 0.16, center + 0.16], [mean, mean], color=color, linewidth=2.2, zorder=5)

    axis.set_title(title)
    axis.set_xlabel(str(treatment_col).replace("_", " ").capitalize())
    axis.set_ylabel(data.snapshot_channel)
    axis.set_xticks(range(len(order)), order, rotation=0)


__all__ = ["render_dual_reporter_triptych"]
