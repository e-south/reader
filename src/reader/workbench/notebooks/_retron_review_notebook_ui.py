from __future__ import annotations

import io
from contextlib import suppress
from dataclasses import replace
from typing import Any

import pandas as pd

from reader.domains.plate_reader.analysis import retron_review_semantics


def retron_figure_label(filename: str) -> str:
    value = str(filename or "").strip()
    if "__sensor=" in value:
        return value.split("__sensor=", 1)[1].replace("_", " ")
    if "__design_id_alias=" in value:
        return value.split("__design_id_alias=", 1)[1].replace("_", " ")
    if "__design_id=" in value:
        return value.split("__design_id=", 1)[1].replace("_", " ")
    return value.replace("_", " ")


def filter_supporting_table_for_figure(table: pd.DataFrame, *, filename: str | None) -> pd.DataFrame:
    if table.empty or not filename:
        return table
    frame = table.copy()
    scope_tokens = {
        "sensor": _scope_token(filename, key="sensor"),
        "design_id_alias": _scope_token(filename, key="design_id_alias"),
        "design_id": _scope_token(filename, key="design_id"),
    }
    for column, token in scope_tokens.items():
        if token is None or column not in frame.columns:
            continue
        mask = frame[column].astype(str).map(retron_review_semantics.slug) == retron_review_semantics.slug(token)
        if not mask.any():
            mask = frame[column].astype(str) == token
        if mask.any():
            frame = frame.loc[mask].copy()
    return frame.reset_index(drop=True)


def retron_notebook_table_preview(
    table: pd.DataFrame | None,
    *,
    max_rows: int = 500,
    max_bytes: int = 350_000,
) -> pd.DataFrame | None:
    if table is None:
        return None
    frame = table.reset_index(drop=True)
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.head(max_rows).copy()
    if max_bytes <= 0 or frame.empty:
        return frame
    preview = frame
    while len(preview) > 1 and _dataframe_csv_bytes(preview) > max_bytes:
        preview = preview.head(max(1, len(preview) // 2)).copy()
    return preview


def figure_to_download_bytes(figure: Any, *, fmt: str) -> bytes:
    buffer = io.BytesIO()
    export_format = str(fmt).lower()
    figure.savefig(
        buffer,
        format=export_format,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
        dpi=240 if export_format == "png" else None,
    )
    return buffer.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def download_safe_stem(value: str) -> str:
    stem = str(value or "").strip()
    if not stem:
        return "retron_review"
    return retron_review_semantics.slug(stem).replace("/", "_") or "retron_review"


def prepare_notebook_plot_figure(item: Any) -> Any:
    fig = getattr(item, "fig", None)
    if fig is None:
        return item
    return replace(item, fig=style_notebook_figure(fig))


def style_notebook_figure(fig: Any) -> Any:
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    with suppress(Exception):
        fig.set_dpi(max(160, int(fig.get_dpi())))
    for axis in getattr(fig, "axes", ()):
        if hasattr(axis, "set_facecolor"):
            axis.set_facecolor("white")
        if hasattr(axis, "tick_params"):
            axis.tick_params(colors="#111111")
        for spine in getattr(axis, "spines", {}).values():
            spine.set_color("#111111")
        if hasattr(axis, "xaxis") and hasattr(axis.xaxis, "label"):
            axis.xaxis.label.set_color("#111111")
        if hasattr(axis, "yaxis") and hasattr(axis.yaxis, "label"):
            axis.yaxis.label.set_color("#111111")
        if hasattr(axis, "title"):
            axis.title.set_color("#111111")
        for text in getattr(axis, "texts", ()):
            text.set_color("#111111")
        if hasattr(axis, "legend"):
            legend = axis.legend_
            if legend is not None:
                _style_notebook_legend(legend)
    for legend in getattr(fig, "legends", ()):
        _style_notebook_legend(legend)
    for text in getattr(fig, "texts", ()):
        text.set_color("#111111")
    return fig


def _style_notebook_legend(legend: Any) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    frame.set_edgecolor("#d0d0d0")
    for text in legend.get_texts():
        text.set_color("#111111")
        with suppress(Exception):
            text.set_fontsize(min(float(text.get_fontsize()), 8.0))
    legend.get_title().set_color("#111111")
    with suppress(Exception):
        legend.get_title().set_fontsize(min(float(legend.get_title().get_fontsize()), 8.0))


def _scope_token(filename: str, *, key: str) -> str | None:
    marker = f"__{key}="
    value = str(filename or "")
    if marker not in value:
        return None
    token = value.split(marker, 1)[1]
    return token.strip() or None


def _dataframe_csv_bytes(df: pd.DataFrame) -> int:
    return len(df.to_csv(index=False).encode("utf-8"))
