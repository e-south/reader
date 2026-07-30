"""Shared plotting helpers for plate-reader renderers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from reader_workbench.plotting.sinks import PlotFigure

from ._data import alias_column, require_columns, warn_if_empty

if TYPE_CHECKING:
    from reader_workbench.plotting.style import PaletteBook


def pretty_name(name: str, suffix: str = "_alias") -> str:
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def best_subplot_grid(n: int) -> tuple[int, int]:
    n = max(1, int(n))
    rows = int(math.floor(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    return rows, cols


def colors_for(n: int, palette_book: PaletteBook | None) -> list[str]:
    if palette_book:
        if n == 1:
            palette = palette_book.colors(2)
            return [palette[1]] if (palette and str(palette[0]).lower() in {"#000000", "black"}) else [palette[0]]
        return palette_book.colors(n)
    from reader_workbench.plotting.style import PaletteBook as _PaletteBook  # noqa: PLC0415

    cycle = _PaletteBook(name="colorblind").colors(max(2, n))
    if n == 1 and str(cycle[0]).lower() in {"#000000", "black"} and len(cycle) > 1:
        return [cycle[1]]
    return cycle[:n]


def plot_figure(
    *,
    fig: Any,
    filename: str,
    fig_kwargs: Mapping[str, Any] | None,
) -> PlotFigure:
    ext = str((fig_kwargs or {}).get("ext", "pdf")).lower()
    dpi = (fig_kwargs or {}).get("dpi", None)
    return PlotFigure(fig=fig, filename=filename, ext=ext, dpi=dpi)


def descriptive_mean_resampling_interval(
    values: np.ndarray,
    *,
    interval_mass: float,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Summarize observed rows without claiming population coverage."""

    mass = _interval_mass(interval_mass)
    draws = _resample_count(resamples)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(finite.mean())
    if finite.size <= 1:
        return mean, mean, mean
    alpha = (1.0 - mass) / 2.0
    if finite.size <= 3:
        resampled_means = finite[_ordered_resampling_indices(finite.size)].mean(axis=1)
    else:
        samples = rng.integers(0, finite.size, size=(draws, finite.size))
        resampled_means = finite[samples].mean(axis=1)
    lower, upper = np.quantile(resampled_means, [alpha, 1.0 - alpha])
    return mean, float(lower), float(upper)


def descriptive_linear_resampling_interval(
    groups: Iterable[np.ndarray | list[float] | tuple[float, ...]],
    *,
    coefficients: Iterable[float],
    interval_mass: float,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    mass = _interval_mass(interval_mass)
    draws = _resample_count(resamples)
    finite_groups: list[np.ndarray] = []
    coeffs = [float(value) for value in coefficients]
    for group in groups:
        values = np.asarray(group, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return (math.nan, math.nan, math.nan)
        finite_groups.append(values)
    if len(finite_groups) != len(coeffs):
        raise ValueError("descriptive_linear_resampling_interval: groups and coefficients must have the same length")

    mean = float(sum(coeff * values.mean() for coeff, values in zip(coeffs, finite_groups, strict=True)))
    if all(values.size <= 1 for values in finite_groups):
        return mean, mean, mean
    alpha = (1.0 - mass) / 2.0

    boot_values = np.zeros(draws, dtype=float)
    for coeff, values in zip(coeffs, finite_groups, strict=True):
        if values.size == 1:
            boot_values += coeff * float(values[0])
            continue
        sample_index = rng.integers(0, values.size, size=(draws, values.size))
        boot_values += coeff * values[sample_index].mean(axis=1)
    lower, upper = np.quantile(boot_values, [alpha, 1.0 - alpha])
    return mean, float(lower), float(upper)


def _interval_mass(value: float) -> float:
    mass = float(value)
    if not math.isfinite(mass) or not 0.0 < mass < 1.0:
        raise ValueError("descriptive interval mass must lie strictly between 0 and 1")
    return mass


def _resample_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("descriptive resample count must be a positive integer")
    return value


def shared_numeric_limits(
    values: Iterable[object],
    *,
    center: float | None = None,
    pad_fraction: float = 0.08,
    min_span: float = 1e-6,
) -> tuple[float, float]:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("shared_numeric_limits: expected at least one finite value")

    lower = float(finite.min())
    upper = float(finite.max())
    span_floor = max(float(min_span), 1e-12)
    pad_fraction = max(0.0, float(pad_fraction))

    if center is not None:
        midpoint = float(center)
        half_span = max(abs(lower - midpoint), abs(upper - midpoint), span_floor / 2.0)
        half_span *= 1.0 + pad_fraction
        return midpoint - half_span, midpoint + half_span

    span = max(upper - lower, span_floor)
    pad = span * pad_fraction
    midpoint = (lower + upper) / 2.0
    half_span = max(span / 2.0 + pad, span_floor / 2.0)
    return midpoint - half_span, midpoint + half_span


@lru_cache(maxsize=4)
def _ordered_resampling_indices(size: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("resampling size must be positive")
    grids = np.indices((size,) * size, dtype=np.intp)
    return grids.reshape(size, -1).T


def annotate_points_smart(
    *,
    ax: Any,
    points: Iterable[tuple[float, float]],
    labels: Iterable[str],
    text_kwargs: Mapping[str, Any] | None = None,
) -> list[Any]:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    point_array = np.asarray(list(points), dtype=float)
    label_list = [str(label) for label in labels]
    if point_array.size == 0 or not label_list:
        return []

    candidate_offsets = [
        (6, 6),
        (6, -6),
        (-6, 6),
        (-6, -6),
        (10, 0),
        (-10, 0),
        (0, 10),
        (0, -10),
        (12, 12),
        (12, -12),
        (-12, 12),
        (-12, -12),
        (18, 0),
        (-18, 0),
        (0, 18),
        (0, -18),
    ]
    display_points = ax.transData.transform(point_array)
    placed_bboxes: list[Any] = []
    annotations: list[Any] = []
    defaults = {
        "fontsize": 8,
        "annotation_clip": False,
        "clip_on": False,
        "zorder": 6,
    }
    defaults.update(dict(text_kwargs or {}))

    for idx, ((x_val, y_val), label) in enumerate(zip(point_array, label_list, strict=False)):
        other_points = np.delete(display_points, idx, axis=0) if len(display_points) > 1 else np.empty((0, 2))
        best_offset = candidate_offsets[0]
        best_score = float("inf")
        for offset in candidate_offsets:
            annotation = ax.annotate(
                label,
                (float(x_val), float(y_val)),
                xytext=offset,
                textcoords="offset points",
                **defaults,
            )
            bbox = annotation.get_window_extent(renderer=renderer).expanded(1.04, 1.10)
            overlap_penalty = sum(_bbox_overlap_area(bbox, existing) for existing in placed_bboxes)
            point_penalty = sum(1.0 for px, py in other_points if bbox.contains(px, py))
            out_penalty = _bbox_outside_axes_penalty(bbox, axes_bbox)
            distance_penalty = abs(offset[0]) + abs(offset[1])
            score = overlap_penalty * 1000.0 + point_penalty * 25000.0 + out_penalty * 25.0 + distance_penalty
            annotation.remove()
            if score < best_score:
                best_score = score
                best_offset = offset
        final_annotation = ax.annotate(
            label,
            (float(x_val), float(y_val)),
            xytext=best_offset,
            textcoords="offset points",
            **defaults,
        )
        placed_bboxes.append(final_annotation.get_window_extent(renderer=renderer).expanded(1.04, 1.10))
        annotations.append(final_annotation)
    return annotations


def _bbox_overlap_area(left: Any, right: Any) -> float:
    if not left.overlaps(right):
        return 0.0
    x_overlap = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    y_overlap = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return float(x_overlap * y_overlap)


def _bbox_outside_axes_penalty(bbox: Any, axes_bbox: Any) -> float:
    penalty = 0.0
    penalty += max(0.0, axes_bbox.x0 - bbox.x0)
    penalty += max(0.0, bbox.x1 - axes_bbox.x1)
    penalty += max(0.0, axes_bbox.y0 - bbox.y0)
    penalty += max(0.0, bbox.y1 - axes_bbox.y1)
    return float(penalty)


__all__ = [
    "alias_column",
    "annotate_points_smart",
    "best_subplot_grid",
    "descriptive_linear_resampling_interval",
    "descriptive_mean_resampling_interval",
    "colors_for",
    "plot_figure",
    "pretty_name",
    "require_columns",
    "shared_numeric_limits",
    "warn_if_empty",
]
