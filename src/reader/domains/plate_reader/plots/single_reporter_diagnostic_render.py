"""Static rendering for the single-reporter diagnostic contract."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from reader.plotting.style import use_style

from .single_reporter_diagnostic import SingleReporterDiagnosticData

_DEFAULT_COLORS = ("#334155", "#0f766e", "#2563eb", "#be123c")
_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")


def render_single_reporter_diagnostic(
    data: SingleReporterDiagnosticData,
    *,
    colors: Sequence[str] | None = None,
    figsize: tuple[float, float] = (14.0, 3.7),
    axis_label_size: float = 9.5,
    tick_label_size: float = 8.5,
    legend_fontsize: float = 8.5,
    line_width: float = 1.7,
    point_size: float = 28.0,
    condition_tick_rotation: float = 20.0,
) -> Figure:
    """Render normalizer, reporter, ratio, and reduced-condition panels."""

    order = list(data.condition_order)
    palette = list(colors or _DEFAULT_COLORS)
    if not palette:
        raise ValueError("single_reporter_diagnostic: colors must contain at least one color")
    color_map = {condition: palette[index % len(palette)] for index, condition in enumerate(order)}
    marker_map = {condition: _MARKERS[index % len(_MARKERS)] for index, condition in enumerate(order)}

    with use_style(
        rc={
            "axes_titleweight": "regular",
            "axes_titlesize": 10.5,
            "axes_labelsize": axis_label_size,
            "font_size": 9.5,
            "legend_fontsize": legend_fontsize,
            "legend_title_fontsize": legend_fontsize,
            "xtick_labelsize": tick_label_size,
            "ytick_labelsize": tick_label_size,
        },
        color_cycle=palette,
    ):
        figure, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
        figure.set_gid("single-reporter-diagnostic")
        for axis, channel in zip(
            axes[:3],
            (data.normalizer_channel, data.reporter_channel, data.ratio_channel),
            strict=True,
        ):
            _draw_kinetics_panel(
                axis,
                data=data,
                channel=channel,
                color_map=color_map,
                marker_map=marker_map,
                line_width=line_width,
                point_size=point_size,
            )
            axis.set_title(f"{channel} kinetics")
            axis.set_xlabel("Time (h)")
            axis.set_ylabel(channel)
            axis.set_box_aspect(1.0)

        qc_axis = _draw_reduction_panel(
            axes[3],
            data=data,
            color_map=color_map,
            marker_map=marker_map,
            point_size=point_size,
            condition_tick_rotation=condition_tick_rotation,
        )
        axes[3].set_box_aspect(1.0)
        qc_axis.set_box_aspect(1.0)

        condition_handles = [
            Line2D(
                [0],
                [0],
                color=color_map[condition],
                marker=marker_map[condition],
                linestyle="-",
                linewidth=line_width,
                markersize=max(4.0, point_size**0.5),
                label=condition,
            )
            for condition in order
        ]
        qc_handle = Line2D(
            [0],
            [0],
            color="#64748b",
            marker="x",
            linestyle="--",
            linewidth=1.1,
            markersize=max(4.0, point_size**0.5),
            label=f"{data.normalizer_channel} QC (right axis)",
        )
        figure.legend(
            handles=[*condition_handles, qc_handle],
            loc="outside lower center",
            ncols=max(1, min(len(condition_handles) + 1, 5)),
            title=str(data.condition_column).replace("_", " ").capitalize(),
        )
        figure.suptitle(f"{data.group_label}\n{data.selection.label}", fontweight="normal")
        return figure


def _draw_kinetics_panel(
    axis,
    *,
    data: SingleReporterDiagnosticData,
    channel: str,
    color_map: dict[str, str],
    marker_map: dict[str, str],
    line_width: float,
    point_size: float,
) -> None:
    channel_rows = data.kinetics[data.kinetics["channel"].astype(str) == str(channel)]
    if channel_rows.empty:
        raise ValueError(f"single_reporter_diagnostic: no kinetics rows for {channel!r}")

    for condition in data.condition_order:
        selected = channel_rows[channel_rows["__condition"].astype(str) == condition]
        if selected.empty:
            continue
        grouped = selected.groupby(["__segment", data.time_column], sort=True, dropna=False)["value"]
        summary = grouped.agg(
            center=lambda values: _center(
                values,
                statistic=data.observation_aggregation.across_unit_statistic,
            ),
            lower=lambda values: float(np.quantile(np.asarray(values, dtype=float), 0.25)),
            upper=lambda values: float(np.quantile(np.asarray(values, dtype=float), 0.75)),
        ).reset_index()
        for _, segment in summary.groupby("__segment", sort=False):
            segment = segment.sort_values(data.time_column)
            x = segment[data.time_column].to_numpy(dtype=float)
            center = segment["center"].to_numpy(dtype=float)
            color = color_map[condition]
            axis.fill_between(
                x,
                segment["lower"].to_numpy(dtype=float),
                segment["upper"].to_numpy(dtype=float),
                color=color,
                alpha=0.14,
                linewidth=0.0,
            )
            axis.plot(x, center, color=color, linewidth=line_width)
            marker_stride = max(1, len(x) // 12)
            axis.scatter(
                x[::marker_stride],
                center[::marker_stride],
                color=color,
                marker=marker_map[condition],
                s=point_size * 0.55,
                linewidths=0.0,
                zorder=3,
            )
    _mark_selection(axis, data=data)


def _mark_selection(axis, *, data: SingleReporterDiagnosticData) -> None:
    if data.selection.endpoint_time_h is not None:
        line = axis.axvline(
            data.selection.endpoint_time_h,
            color="#111827",
            linewidth=1.0,
            linestyle=(0, (5, 3)),
            zorder=0.8,
        )
        line.set_gid("single-reporter-endpoint")
        return
    assert data.selection.window_h is not None
    lower, upper = data.selection.window_h
    patch = axis.axvspan(lower, upper, color="#d8b365", alpha=0.18, linewidth=0.0, zorder=0.5)
    patch.set_gid("single-reporter-window")


def _draw_reduction_panel(
    axis,
    *,
    data: SingleReporterDiagnosticData,
    color_map: dict[str, str],
    marker_map: dict[str, str],
    point_size: float,
    condition_tick_rotation: float,
):
    qc_axis = axis.twinx()
    qc_axis.set_gid("single-reporter-normalizer-qc")
    qc_axis.grid(False)
    qc_axis.spines["right"].set_visible(True)
    qc_axis.spines["right"].set_color("#64748b")
    qc_axis.tick_params(axis="y", colors="#64748b")

    for index, condition in enumerate(data.condition_order):
        ratio = data.reduced_ratio[data.reduced_ratio["__condition"].astype(str) == condition]
        normalizer = data.reduced_normalizer[data.reduced_normalizer["__condition"].astype(str) == condition]
        ratio_values = ratio["value"].to_numpy(dtype=float)
        normalizer_values = normalizer["value"].to_numpy(dtype=float)
        ratio_offsets = _offsets(len(ratio_values), center=-0.07)
        normalizer_offsets = _offsets(len(normalizer_values), center=0.07)
        color = color_map[condition]
        axis.scatter(
            index + ratio_offsets,
            ratio_values,
            s=point_size,
            facecolors="white",
            edgecolors=color,
            marker=marker_map[condition],
            linewidths=1.1,
            zorder=3,
        )
        ratio_center = _center(
            ratio_values,
            statistic=data.observation_aggregation.across_unit_statistic,
        )
        ratio_line = axis.hlines(ratio_center, index - 0.22, index + 0.02, color=color, linewidth=2.4, zorder=4)
        ratio_line.set_gid("single-reporter-ratio-center")

        qc_axis.scatter(
            index + normalizer_offsets,
            normalizer_values,
            s=point_size * 0.9,
            color="#64748b",
            marker="x",
            linewidths=1.1,
            zorder=3,
        )
        normalizer_center = _center(
            normalizer_values,
            statistic=data.observation_aggregation.across_unit_statistic,
        )
        qc_line = qc_axis.hlines(
            normalizer_center,
            index + 0.02,
            index + 0.22,
            color="#64748b",
            linewidth=1.5,
            linestyle="--",
            zorder=4,
        )
        qc_line.set_gid("single-reporter-normalizer-center")

    axis.set_title(f"{data.ratio_channel} by condition")
    axis.set_xlabel(str(data.condition_column).replace("_", " ").capitalize())
    value_space = data.selection.temporal_reduction.output_space
    value_suffix = "" if value_space == "linear" else " (log2)"
    axis.set_ylabel(f"Reduced {data.ratio_channel}{value_suffix}")
    qc_axis.set_ylabel(f"Reduced {data.normalizer_channel}{value_suffix} (QC only)", color="#64748b")
    axis.set_xticks(
        range(len(data.condition_order)),
        data.condition_order,
        rotation=condition_tick_rotation,
        ha="right" if condition_tick_rotation else "center",
    )
    axis.set_xlim(-0.45, max(0.55, len(data.condition_order) - 0.55))
    return qc_axis


def _offsets(size: int, *, center: float) -> np.ndarray:
    if size <= 1:
        return np.asarray([center] if size else [], dtype=float)
    return np.linspace(center - 0.06, center + 0.06, size)


def _center(values, *, statistic: str) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("single_reporter_diagnostic: reduction contains no finite values")
    if statistic == "mean":
        return float(finite.mean())
    return float(np.median(finite))


__all__ = ["render_single_reporter_diagnostic"]
