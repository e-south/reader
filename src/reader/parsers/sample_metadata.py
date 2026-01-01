"""
--------------------------------------------------------------------------------
<reader project>
src/reader/parsers/sample_metadata.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pandas as pd


def parse_sample_metadata(path: str, *, key: str = "sample_id") -> pd.DataFrame:
    """
    Load a sample metadata table keyed by `sample_id` (case-insensitive).
    Accepts CSV/TSV or Excel. Column names are not coerced except for the key.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    df = pd.read_excel(p) if suffix in {".xls", ".xlsx"} else pd.read_csv(p)

    cols = {c.lower(): c for c in df.columns}
    if key.lower() not in cols:
        raise ValueError(f"Metadata must contain a '{key}' column")
    if cols[key.lower()] != key:
        df = df.rename(columns={cols[key.lower()]: key})
    return df
