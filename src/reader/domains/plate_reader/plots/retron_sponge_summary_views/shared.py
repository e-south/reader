from __future__ import annotations

import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from reader.plotting.style import PaletteBook

from ..common import colors_for

_FAMILY_ORDER = {"mono": 0, "bi": 1, "tri": 2, "quad": 3, "control": 4}


@dataclass(frozen=True)
class _SummarySubplotPolicy:
    top: float
    bottom: float
    left: float
    right: float
    hspace: float
    wspace: float


@dataclass(frozen=True)
class _SummaryFigurePolicy:
    default_figsize: tuple[float, float]
    title_y: float
    subtitle_y: float
    adjust: _SummarySubplotPolicy
    xlabel: str | None = None
    xlabel_y: float | None = None
    xlabel_fontsize: float = 11.0
    ylabel: str | None = None
    ylabel_x: float | None = None
    ylabel_fontsize: float = 11.0


@dataclass(frozen=True)
class _RetronSummaryPlotRequest:
    summary: pd.DataFrame
    trace: pd.DataFrame | None
    output_dir: Path | None
    title: str
    filename: str | None
    palette_book: PaletteBook | None
    control_name: str
    no_stress_label: str
    relevant_only: bool
    metric: str | None
    state_order: Sequence[str] | None
    burden_metric: str
    fig_kwargs: dict


def _level_color_map(levels: Sequence[str], *, palette_book: PaletteBook | None) -> dict[str, str]:
    color_values = colors_for(max(1, len(levels)), palette_book)
    return {str(level): color_values[idx % len(color_values)] for idx, level in enumerate(levels)}


def _summary_figure_size(
    *,
    fig_kwargs: Mapping[str, object],
    policy: _SummaryFigurePolicy,
) -> tuple[float, float]:
    figsize = fig_kwargs.get("figsize", policy.default_figsize)
    return float(fig_kwargs.get("figsize", policy.default_figsize)[0]), float(figsize[1])


def _new_summary_grid_figure(
    *,
    rows: int,
    cols: int,
    policy: _SummaryFigurePolicy,
    fig_kwargs: Mapping[str, object],
    sharex: bool | str = False,
    sharey: bool | str = False,
    gridspec_kw: Mapping[str, object] | None = None,
):
    width, height = _summary_figure_size(fig_kwargs=fig_kwargs, policy=policy)
    fig = Figure(figsize=(width, height), constrained_layout=False)
    FigureCanvasAgg(fig)
    axes = fig.subplots(
        nrows=rows,
        ncols=cols,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=dict(gridspec_kw or {}),
    )
    return fig, axes


def _finalize_summary_figure(
    fig,
    *,
    policy: _SummaryFigurePolicy,
    fig_kwargs: Mapping[str, object],
    title: str,
    subtitle: str,
    context: str | None = None,
) -> None:
    _set_figure_header(
        fig,
        title=title,
        context=context,
        subtitle=subtitle,
        title_y=float(fig_kwargs.get("suptitle_y", policy.title_y)),
        subtitle_y=float(fig_kwargs.get("subtitle_y", policy.subtitle_y)),
    )
    if policy.xlabel and policy.xlabel_y is not None:
        fig.supxlabel(policy.xlabel, y=policy.xlabel_y, fontsize=policy.xlabel_fontsize)
    if policy.ylabel and policy.ylabel_x is not None:
        fig.supylabel(policy.ylabel, x=policy.ylabel_x, fontsize=policy.ylabel_fontsize)
    fig.subplots_adjust(
        top=policy.adjust.top,
        bottom=policy.adjust.bottom,
        left=policy.adjust.left,
        right=policy.adjust.right,
        hspace=policy.adjust.hspace,
        wspace=policy.adjust.wspace,
    )


def _pivot_summary(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(index="sensor", columns="sponge", values="value", aggfunc="mean")
    if table.empty:
        return table
    row_order = _ordered(table.index.tolist())
    col_order = sorted(table.columns, key=_sponge_sort_key)
    return table.reindex(index=row_order, columns=col_order)


def _set_axis_title(ax, title: str, *, pad: float = 8.0) -> None:
    ax.set_title(_wrap_plot_text(str(title), width=24), pad=pad, fontweight="normal", fontsize=10)


def _set_figure_header(
    fig,
    *,
    title: str,
    context: str | None = None,
    subtitle: str | None = None,
    title_y: float = 0.98,
    subtitle_y: float = 0.945,
) -> None:
    _set_figure_title(fig, title=title, context=context, y=title_y)
    subtitle_text = str(subtitle or "").strip()
    if subtitle_text:
        fig.text(
            0.5,
            subtitle_y,
            _wrap_plot_text(subtitle_text, width=108),
            ha="center",
            va="top",
            color="#333333",
            fontsize=10.4,
        )


def _set_figure_title(fig, *, title: str, context: str | None = None, y: float = 0.98) -> None:
    figure_title = str(title).strip()
    context_text = str(context or "").strip()
    if context_text:
        figure_title = f"{figure_title} · {context_text}"
    fig.suptitle(_wrap_plot_text(figure_title, width=86), y=y, x=0.5, ha="center", fontweight="normal", fontsize=14.4)


def _wrap_plot_text(text: str, *, width: int) -> str:
    value = str(text or "").strip()
    if not value or len(value) <= width:
        return value
    if "\n" in value:
        return "\n".join(_wrap_plot_text(line, width=width) for line in value.splitlines())
    parts = [part.strip() for part in value.split(";") if part.strip()]
    if len(parts) > 1:
        lines: list[str] = []
        current = ""
        for part in parts:
            segment = part if not current else f"{current}; {part}"
            if len(segment) <= width:
                current = segment
                continue
            if current:
                lines.append(current)
            current = part
        if current:
            lines.append(current)
        return "\n".join(lines)
    return textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)


def _wrap_hyphenated_label(label: str, *, max_parts_per_line: int = 2) -> str:
    parts = [part for part in str(label).split("-") if part]
    if len(parts) <= max_parts_per_line:
        return str(label)
    lines = ["-".join(parts[index : index + max_parts_per_line]) for index in range(0, len(parts), max_parts_per_line)]
    return "\n".join(lines)


def _first_non_null(series: pd.Series) -> str:
    for value in series:
        if pd.notna(value):
            return str(value)
    return "other"


def _auc(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) == 0 or len(values) == 0:
        return float("nan")
    finite = np.isfinite(times) & np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.trapezoid(values[finite], times[finite]))


def _require_relevant_sensor_pair(df: pd.DataFrame, *, where: str) -> None:
    if "relevant_sensor_pair" not in df.columns:
        raise ValueError(f"{where}: relevant_sensor_pair is required for on-target filtering")


def _preferred_stresses(values: Iterable[object], *, stress_order: Sequence[str] | None) -> list[str]:
    preferred = [str(value) for value in (stress_order or []) if str(value).strip()]
    return _ordered(values, preferred=preferred or ("H2O",))


def _sponge_levels(df: pd.DataFrame, *, control_name: str) -> list[str]:
    levels = df["sponge"].dropna().astype(str).unique().tolist()
    return sorted(levels, key=lambda item: _sponge_sort_key(item, control_name=control_name))


def _sponge_sort_key(value: str, *, control_name: str = "tetO") -> tuple[int, str]:
    if value == control_name:
        return (_FAMILY_ORDER["control"], value)
    parts = [part for part in str(value).split("-") if part]
    size = {1: "mono", 2: "bi", 3: "tri", 4: "quad"}.get(len(parts), "other")
    return (_FAMILY_ORDER.get(size, 99), str(value))


def _ordered(values: Iterable[object], preferred: Sequence[str] | None = None) -> list[str]:
    seen = {str(value) for value in values if pd.notna(value)}
    ordered = [item for item in (preferred or []) if item in seen]
    ordered.extend(sorted(item for item in seen if item not in ordered))
    return ordered


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value)).strip("_").lower()
