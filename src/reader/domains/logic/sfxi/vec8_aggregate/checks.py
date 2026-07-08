from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from reader.errors import SFXIError

from .constants import REQUIRED_VEC8_COLUMNS


def require_vec8_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_VEC8_COLUMNS if column not in frame.columns]
    if missing:
        raise SFXIError(f"SFXI vec8 aggregate requires vec8 columns: {', '.join(missing)}.")


def require_normalized_frame(frame: pd.DataFrame) -> None:
    require_vec8_columns(frame)
    missing = [column for column in ("source_id", "row_label") if column not in frame.columns]
    if missing:
        raise SFXIError(f"SFXI vec8 aggregate requires normalized columns: {', '.join(missing)}.")
    if frame.empty:
        raise SFXIError("SFXI vec8 aggregate has no rows to plot.")


def finite_numeric_column(series: pd.Series, *, column: str, source: Path) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() | ~values.map(lambda value: math.isfinite(float(value)))
    if invalid.any():
        raise SFXIError(f"SFXI vec8 aggregate column {column!r} must be finite numeric values in {source}.")
    return values.astype(float)
