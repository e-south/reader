from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from reader.errors import SFXIError

from ..validation import require_intensity_delta_column
from .constants import REQUIRED_VEC8_COLUMNS


def require_vec8_columns(frame: pd.DataFrame) -> None:
    require_intensity_delta_column(frame)
    missing = [column for column in REQUIRED_VEC8_COLUMNS if column not in frame.columns]
    if missing:
        raise SFXIError(f"SFXI vec8 input requires columns: {', '.join(missing)}.")


def require_normalized_frame(frame: pd.DataFrame) -> None:
    require_vec8_columns(frame)
    missing = [column for column in ("source_id", "row_label") if column not in frame.columns]
    if missing:
        raise SFXIError(f"SFXI vec8 aggregate requires normalized columns: {', '.join(missing)}.")
    if frame.empty:
        raise SFXIError("SFXI vec8 aggregate has no rows to plot.")


def finite_numeric_column(
    series: pd.Series,
    *,
    column: str,
    source: Path,
    allow_nan: bool = False,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    coerced_to_nan = series.notna() & values.isna()
    nonfinite = values.notna() & ~values.fillna(0.0).map(lambda value: math.isfinite(float(value)))
    invalid = coerced_to_nan | nonfinite
    if not allow_nan:
        invalid |= values.isna()
    if invalid.any():
        raise SFXIError(f"SFXI vec8 aggregate column {column!r} must be finite numeric values in {source}.")
    return values.astype(float)
