"""Plate-reader sample-map parsing."""

from collections import Counter
from pathlib import Path

import pandas as pd


def parse_sample_map(path: str | Path) -> pd.DataFrame:
    """
    Load a sample metadata map with a 'position' key.

    Accepted inputs:
      1) A table with an explicit 'position' column (preferred).
      2) A table with 'row' and 'col' columns (will be combined into 'position').

    This parser is plate-reader oriented: the join key is a well position.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Sample map does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Sample map must be a regular file: {p}")
    suffix = p.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(p)
    elif suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported sample-map format {suffix or '<none>'!r}; expected .csv or .xlsx")

    normalized_columns = [str(column).strip().lower() for column in df.columns]
    duplicate_columns = sorted(name for name, count in Counter(normalized_columns).items() if count > 1)
    if duplicate_columns:
        raise ValueError(f"Sample map has duplicate column names after normalization: {duplicate_columns}")
    cols = dict(zip(normalized_columns, df.columns, strict=True))
    if "position" in cols:
        if cols["position"] != "position":
            df = df.rename(columns={cols["position"]: "position"})
    elif {"row", "col"}.issubset(cols):
        row_col = cols["row"]
        col_col = cols["col"]
        out = df.copy()
        row_values = out[row_col].astype("string").str.strip()
        col_values = out[col_col].astype("string").str.strip()
        invalid_parts = row_values.isna() | row_values.eq("") | col_values.isna() | col_values.eq("")
        if invalid_parts.any():
            rows = [int(index) + 2 for index in out.index[invalid_parts].tolist()]
            raise ValueError(f"Sample map has blank row/col values at file rows: {rows}")
        out["position"] = row_values + col_values
        df = out.drop(columns=[row_col, col_col])
    else:
        raise ValueError("Sample map must contain either a 'position' column or a ('row','col') pair.")

    positions = df["position"].astype("string").str.strip().str.upper()
    invalid = positions.isna() | positions.eq("") | positions.eq("NAN") | positions.eq("<NA>")
    if invalid.any():
        rows = [int(index) + 2 for index in df.index[invalid].tolist()]
        raise ValueError(f"Sample map has blank position values at file rows: {rows}")
    duplicates = sorted(positions[positions.duplicated(keep=False)].unique().tolist())
    if duplicates:
        raise ValueError(f"Sample map positions must be unique; duplicates: {duplicates}")
    df = df.copy()
    df["position"] = positions
    return df
