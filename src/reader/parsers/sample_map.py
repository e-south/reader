"""
--------------------------------------------------------------------------------
<reader project>
src/reader/parsers/sample_map.py

Author(s): Eric J. South
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

    The function is instrument-agnostic. 'position' can be any join key that matches
    your measurement table.
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
