from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from reader.errors import SFXIError

from .vec8_aggregate.checks import require_vec8_columns
from .vec8_aggregate.constants import VEC8_CHANNELS
from .vec8_aggregate.render import render_sfxi_vec8_heatmap

_NUMERIC_COLUMNS = (
    "time_selected_h",
    "intensity_log2_offset_delta",
    "r_logic",
    *VEC8_CHANNELS,
)


def normalize_experiment_vec8_heatmap_frame(vec8: pd.DataFrame, *, source_id: str) -> pd.DataFrame:
    """Adapt one experiment's sfxi.vec8.v3 table to the heatmap renderer contract."""

    require_vec8_columns(vec8)
    if vec8.empty:
        raise SFXIError("SFXI vec8 heatmap has no rows to plot.")
    normalized_source_id = _non_empty_text(source_id, field="source_id")

    frame = vec8.copy().reset_index(drop=True)
    _require_non_empty_text_columns(frame, columns=("design_id", "reference_design_id"))
    for column in _NUMERIC_COLUMNS:
        frame[column] = _finite_numeric(frame[column], column=column)
    frame["flat_logic"] = frame["flat_logic"].astype(bool)

    frame["source_index"] = 0
    frame["source_id"] = normalized_source_id
    frame["source_kind"] = "experiment_record"
    frame["source_row_index"] = range(len(frame))
    frame["row_label"] = _row_labels(frame, source_id=normalized_source_id)
    return frame


def render_experiment_sfxi_vec8_heatmap(
    vec8: pd.DataFrame,
    *,
    source_id: str,
    title: str | None = None,
    max_y_tick_labels: int = 80,
):
    frame = normalize_experiment_vec8_heatmap_frame(vec8, source_id=source_id)
    return render_sfxi_vec8_heatmap(frame, title=title, max_y_tick_labels=max_y_tick_labels)


def _row_labels(frame: pd.DataFrame, *, source_id: str) -> list[str]:
    return [f"{source_id}::{design_id}" for design_id in frame["design_id"].astype(str).tolist()]


def _finite_numeric(series: pd.Series, *, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() | ~values.map(lambda value: math.isfinite(float(value)))
    if invalid.any():
        raise SFXIError(f"SFXI vec8 heatmap column {column!r} must contain finite numeric values.")
    return values.astype(float)


def _require_non_empty_text_columns(frame: pd.DataFrame, *, columns: Sequence[str]) -> None:
    for column in columns:
        values = frame[column].astype(str).str.strip()
        if values.eq("").any():
            raise SFXIError(f"SFXI vec8 heatmap column {column!r} must contain non-empty values.")


def _non_empty_text(value: str, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise SFXIError(f"SFXI vec8 heatmap {field} must be a non-empty string.")
    return text
