from __future__ import annotations

import math
import re
import textwrap

import numpy as np
import pandas as pd

from reader_workbench.errors import SFXIError
from reader_workbench.plotting.style import use_style

from .checks import require_normalized_frame
from .constants import VEC8_CHANNELS

CHANNEL_LABELS = ("v00", "v10", "v01", "v11", "y00*", "y10*", "y01*", "y11*")
LOGIC_CHANNEL_COUNT = 4
LOGIC_COLORBAR_LABEL = "$v_i$ normalized response"
INTENSITY_COLORBAR_LABEL = "$y_i^\\star$ anchored log2 intensity"
NATURAL_SORT_TOKEN = re.compile(r"\d+|\D+")
TILE_SIZE_IN = 0.38
AXES_LEFT_MAX = 0.52
AXES_RIGHT = 0.94
AXES_BOTTOM = 0.24
AXES_TOP = 0.84
MIN_TICK_FONT_SIZE = 6.8
MAX_ROW_TICK_FONT_SIZE = 10.5
MAX_CHANNEL_TICK_FONT_SIZE = 12.5


def render_sfxi_vec8_heatmap(
    frame: pd.DataFrame,
    *,
    title: str | None = None,
    max_y_tick_labels: int = 80,
):
    require_normalized_frame(frame)
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - dependency guard
        raise SFXIError("SFXI vec8 aggregate heatmap requires matplotlib.") from exc

    plot_frame = _ordered_plot_frame(frame)
    values = plot_frame.loc[:, list(VEC8_CHANNELS)].astype(float)
    row_labels = _display_row_labels(plot_frame)
    row_count = len(values)
    matrix = values.to_numpy()
    figsize = _figure_size(row_labels, row_count=row_count)
    row_tick_font_size = _row_tick_font_size(row_count=row_count, figure_height=figsize[1])
    channel_tick_font_size = _channel_tick_font_size(figure_width=figsize[0])
    visible_y_tick_labels = _visible_y_tick_label_count(
        max_labels=max_y_tick_labels,
        row_count=row_count,
        figure_height=figsize[1],
        font_size=row_tick_font_size,
    )
    annotation_font_size = _annotation_font_size(figure_width=figsize[0])
    colorbar_font_size = _colorbar_font_size(row_tick_font_size)
    with use_style(
        {
            "figure_figsize": figsize,
            "axes_grid": False,
            "xtick_labelsize": channel_tick_font_size,
            "ytick_labelsize": row_tick_font_size,
            "axes_titlesize": annotation_font_size,
            "font_size": 12.0,
        }
    ):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
        fig.subplots_adjust(
            left=_left_margin(row_labels, figure_width=figsize[0]),
            right=AXES_RIGHT,
            bottom=AXES_BOTTOM,
            top=AXES_TOP,
        )

        x_edges = np.arange(len(VEC8_CHANNELS) + 1)
        y_edges = np.arange(row_count + 1)
        logic_mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            _masked_channel_block(matrix, start=0, stop=LOGIC_CHANNEL_COUNT),
            cmap=_logic_colormap(LinearSegmentedColormap),
            norm=_logic_norm(matrix[:, :LOGIC_CHANNEL_COUNT], Normalize),
            edgecolors="white",
            linewidth=0.65,
            shading="flat",
        )
        intensity_mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            _masked_channel_block(matrix, start=LOGIC_CHANNEL_COUNT, stop=len(VEC8_CHANNELS)),
            cmap=_intensity_colormap(LinearSegmentedColormap),
            norm=_centered_norm(matrix[:, LOGIC_CHANNEL_COUNT:], TwoSlopeNorm),
            edgecolors="white",
            linewidth=0.65,
            shading="flat",
        )

        ax.set_xlim(0, len(VEC8_CHANNELS))
        ax.set_ylim(row_count, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("NW")
        ax.set_ylabel("source :: design")
        ax.set_xticks(np.arange(len(VEC8_CHANNELS)) + 0.5)
        ax.set_xticklabels(CHANNEL_LABELS, rotation=90, ha="center", va="top", fontsize=channel_tick_font_size)
        _set_y_ticks(ax, row_labels, max_labels=visible_y_tick_labels, fontsize=row_tick_font_size)
        _draw_channel_annotations(ax, font_size=annotation_font_size)
        _draw_row_group_boundaries(ax, plot_frame["design_id"].astype(str).tolist())
        _draw_heatmap_centered_title(fig, ax, _wrapped_title(title or "SFXI vec8 aggregate"))
        _draw_split_colorbars(fig, ax, logic_mesh, intensity_mesh, font_size=colorbar_font_size)
        ax.tick_params(axis="both", length=0)
        ax.axvline(LOGIC_CHANNEL_COUNT, color="white", linewidth=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    return fig


def _figure_size(labels: list[str], *, row_count: int) -> tuple[float, float]:
    label_width = _label_width(labels)
    heatmap_width = len(VEC8_CHANNELS) * TILE_SIZE_IN
    side_width = 1.55
    width = max(7.2, min(11.4, label_width + heatmap_width + side_width))
    height = max(5.4, min(20.0, row_count * TILE_SIZE_IN + 3.05))
    return (width, height)


def _label_width(labels: list[str]) -> float:
    max_label_len = max((len(label) for label in labels), default=0)
    return max(3.0, min(4.6, 2.55 + 0.045 * max_label_len))


def _left_margin(labels: list[str], *, figure_width: float) -> float:
    return min(AXES_LEFT_MAX, _label_width(labels) / figure_width)


def _axis_height_inches(figure_height: float) -> float:
    return max(1.0, float(figure_height) * (AXES_TOP - AXES_BOTTOM))


def _row_tick_font_size(*, row_count: int, figure_height: float) -> float:
    row_pitch_points = _axis_height_inches(figure_height) * 72.0 / max(int(row_count), 1)
    return max(MIN_TICK_FONT_SIZE, min(MAX_ROW_TICK_FONT_SIZE, row_pitch_points * 0.78))


def _channel_tick_font_size(*, figure_width: float) -> float:
    channel_pitch_points = (
        float(figure_width) * (AXES_RIGHT - _left_margin([], figure_width=figure_width)) * 72.0
    ) / len(VEC8_CHANNELS)
    return max(10.5, min(MAX_CHANNEL_TICK_FONT_SIZE, channel_pitch_points * 0.34))


def _visible_y_tick_label_count(*, max_labels: int, row_count: int, figure_height: float, font_size: float) -> int:
    axis_points = _axis_height_inches(figure_height) * 72.0
    capacity = max(1, math.floor(axis_points / (float(font_size) * 1.22)))
    return max(1, min(int(max_labels), int(row_count), capacity))


def _annotation_font_size(*, figure_width: float) -> float:
    return max(12.0, min(14.0, float(figure_width) * 1.10))


def _colorbar_font_size(row_tick_font_size: float) -> float:
    return max(8.5, min(10.2, float(row_tick_font_size)))


def _wrapped_title(title: str) -> str:
    return textwrap.fill(str(title), width=72)


def _set_y_ticks(ax, labels: list[str], *, max_labels: int, fontsize: float) -> None:
    n_rows = len(labels)
    if n_rows <= max_labels:
        ticks = list(range(n_rows))
    else:
        step = max(1, math.ceil(n_rows / max_labels))
        ticks = list(range(0, n_rows, step))
        if ticks[-1] != n_rows - 1:
            ticks.append(n_rows - 1)
    ax.set_yticks([tick + 0.5 for tick in ticks])
    ax.set_yticklabels([labels[index] for index in ticks], fontsize=fontsize)


def _draw_row_group_boundaries(ax, design_ids: list[str]) -> None:
    previous = _row_group_key(design_ids[0])
    for index, design_id in enumerate(design_ids[1:], start=1):
        group = _row_group_key(design_id)
        if group != previous:
            ax.axhline(index, color="#ffffff", linewidth=2.0)
            ax.axhline(index, color="#4f4f4f", linewidth=0.55, alpha=0.45)
            previous = group


def _masked_channel_block(matrix: np.ndarray, *, start: int, stop: int) -> np.ma.MaskedArray:
    mask = np.ones(matrix.shape, dtype=bool)
    mask[:, start:stop] = False
    return np.ma.masked_array(matrix, mask=mask)


def _logic_norm(values: np.ndarray, normalize_type):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    upper = max(1.0, float(finite.max())) if finite.size else 1.0
    lower = min(0.0, float(finite.min())) if finite.size else 0.0
    return normalize_type(vmin=lower, vmax=upper)


def _centered_norm(values: np.ndarray, norm_type):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    limit = max(1.0, float(np.abs(finite).max())) if finite.size else 1.0
    return norm_type(vmin=-limit, vcenter=0.0, vmax=limit)


def _logic_colormap(colormap_type):
    cmap = colormap_type.from_list("sfxi_logic_blue", ("#f7f9fb", "#9bbfda", "#10306d"))
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def _intensity_colormap(colormap_type):
    cmap = colormap_type.from_list("sfxi_intensity_diverging", ("#10306d", "#f7f7f2", "#c96845"))
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def _draw_channel_annotations(ax, *, font_size: float) -> None:
    transform = ax.get_xaxis_transform()
    ax.text(
        LOGIC_CHANNEL_COUNT / 2,
        1.13,
        "Logic\npattern",
        ha="center",
        va="bottom",
        linespacing=0.9,
        fontsize=font_size,
        transform=transform,
        clip_on=False,
    )
    ax.text(
        LOGIC_CHANNEL_COUNT + (len(VEC8_CHANNELS) - LOGIC_CHANNEL_COUNT) / 2,
        1.13,
        "Anchored\nintensity",
        ha="center",
        va="bottom",
        linespacing=0.9,
        fontsize=font_size,
        transform=transform,
        clip_on=False,
    )
    for start, stop in ((0, LOGIC_CHANNEL_COUNT), (LOGIC_CHANNEL_COUNT, len(VEC8_CHANNELS))):
        ax.plot(
            [start + 0.08, stop - 0.08],
            [1.08, 1.08],
            color="#8a8a8a",
            linewidth=1.6,
            transform=transform,
            clip_on=False,
        )
    ax.text(
        len(VEC8_CHANNELS) / 2,
        1.04,
        r"vec8 = concat($v$, $y^\star$)",
        ha="center",
        va="top",
        fontsize=max(11.0, font_size - 3.0),
        transform=transform,
        clip_on=False,
    )


def _draw_heatmap_centered_title(fig, ax, title: str) -> None:
    fig.canvas.draw()
    box = ax.get_position()
    fig.text(
        (box.x0 + box.x1) / 2,
        0.965,
        title,
        ha="center",
        va="top",
        fontsize=15.0,
    )


def _draw_split_colorbars(fig, ax, logic_mesh, intensity_mesh, *, font_size: float) -> None:
    fig.canvas.draw()
    box = ax.get_position()
    colorbar_height = 0.014
    # Anchor colorbars in the reserved bottom margin. Sparse heatmaps can shift
    # the aspect-adjusted axes upward, which should not move the legend stack.
    logic_y = AXES_BOTTOM - 0.070
    intensity_y = max(0.035, logic_y - 0.055)
    logic_cax = fig.add_axes([box.x0, logic_y, box.width, colorbar_height])
    intensity_cax = fig.add_axes([box.x0, intensity_y, box.width, colorbar_height])
    logic_cbar = fig.colorbar(logic_mesh, cax=logic_cax, orientation="horizontal")
    logic_cbar.ax.xaxis.set_label_position("top")
    logic_cbar.ax.xaxis.set_ticks_position("top")
    logic_cbar.set_label(LOGIC_COLORBAR_LABEL, labelpad=4)
    logic_cbar.set_ticks([0.0, 1.0])
    intensity_cbar = fig.colorbar(intensity_mesh, cax=intensity_cax, orientation="horizontal")
    intensity_cbar.set_label(INTENSITY_COLORBAR_LABEL, labelpad=4)
    for colorbar in (logic_cbar, intensity_cbar):
        colorbar.ax.tick_params(axis="x", labelsize=font_size, length=2)
        colorbar.ax.xaxis.label.set_size(font_size)


def _ordered_plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    order = sorted(range(len(frame)), key=lambda index: _row_sort_key(frame.iloc[index], fallback_index=index))
    return frame.iloc[order].reset_index(drop=True)


def _row_sort_key(row: pd.Series, *, fallback_index: int) -> tuple[object, ...]:
    design_id = str(row["design_id"])
    source_label = str(row[_source_label_column(row.index)])
    return (_natural_sort_key(design_id), _natural_sort_key(source_label), fallback_index)


def _row_group_key(design_id: str) -> tuple[object, ...]:
    return _natural_sort_key(_design_family_label(design_id))


def _display_row_labels(frame: pd.DataFrame) -> list[str]:
    source_labels = frame[_source_label_column(frame.columns)].astype(str).map(_short_source_label)
    design_labels = frame["design_id"].astype(str).map(_short_design_label)
    time_labels = frame["time_selected_h"].map(_timepoint_label) if "time_selected_h" in frame.columns else None
    labels = []
    for index, source_label in enumerate(source_labels.tolist()):
        time_label = "" if time_labels is None else f" {time_labels.iloc[index]}"
        labels.append(f"{source_label}{time_label} :: {design_labels.iloc[index]}")
    if len(set(labels)) == len(labels):
        return labels
    fallback = frame["row_label"].astype(str)
    if time_labels is not None:
        fallback = fallback + " @ " + time_labels.astype(str)
    return fallback.tolist()


def _source_label_column(columns) -> str:
    if "source_resource_id" in columns:
        return "source_resource_id"
    if "source_experiment_id" in columns:
        return "source_experiment_id"
    raise SFXIError("SFXI vec8 heatmap requires explicit source resource or experiment identity.")


def _short_source_label(value: str) -> str:
    return _middle_truncate(value.strip(), max_chars=24)


def _short_design_label(value: str) -> str:
    return _middle_truncate(value.strip(), max_chars=28)


def _design_family_label(design_id: str) -> str:
    text = design_id.strip()
    match = re.match(r"[A-Za-z]+", text)
    return match.group(0) if match else text


def _natural_sort_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token.lower()) for token in NATURAL_SORT_TOKEN.findall(value)
    )


def _timepoint_label(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"t={value}"
    if not math.isfinite(number):
        return f"t={value}"
    return f"t={number:.2f}h"


def _middle_truncate(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    keep = max_chars - 3
    head = keep // 2
    tail = keep - head
    return f"{value[:head]}...{value[-tail:]}"
