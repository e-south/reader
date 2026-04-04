"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/io/sample_map.py

Plate-reader sample-map parsing.
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pandas as pd


def parse_sample_map(path: str) -> pd.DataFrame:
    """
    Load a sample metadata map with a 'position' key.

    Accepted inputs:
      1) A table with an explicit 'position' column (preferred).
      2) A table with 'row' and 'col' columns (will be combined into 'position').

    This parser is plate-reader oriented: the join key is a well position.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    # Load file
    df = pd.read_excel(p) if suffix in {".xls", ".xlsx"} else pd.read_csv(p)

    cols = {c.lower(): c for c in df.columns}
    if "position" in cols:
        # Normalize column name to exactly 'position' if case differs
        if cols["position"] != "position":
            df = df.rename(columns={cols["position"]: "position"})
        return df

    # Back-up: build position from row/col if provided
    if {"row", "col"}.issubset(cols):
        row_col = cols["row"]
        col_col = cols["col"]
        out = df.copy()
        out["position"] = out[row_col].astype(str).str.strip() + out[col_col].astype(str).str.strip()
        return out.drop(columns=[row_col, col_col])

    raise ValueError("Sample map must contain either a 'position' column or a ('row','col') pair.")
