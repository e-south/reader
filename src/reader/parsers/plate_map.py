"""
--------------------------------------------------------------------------------
<reader project>

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path
import pandas as pd


def parse_plate_map(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()

    # 1) Load file
    if suffix in ('.xls', '.xlsx'):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    # 2) Validate presence of row & col
    if 'row' not in df.columns or 'col' not in df.columns:
        raise ValueError("Plate map must have 'row' and 'col' columns")

    # 3) Clean and construct 'position'
    df['row'] = df['row'].astype(str).str.strip()
    df['col'] = df['col'].astype(str).str.strip()
    df['position'] = df['row'] + df['col']

    # 4) Drop helpers and return
    return df.drop(['row', 'col'], axis=1)