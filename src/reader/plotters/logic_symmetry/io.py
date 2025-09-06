"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plotters/logic_symmetry/io.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_plot(fig, output_dir: Path, base_name: str, formats: List[str], dpi: int) -> List[Path]:
    ensure_dir(output_dir)
    written: List[Path] = []
    for ext in formats:
        path = output_dir / f"{base_name}.{ext.lower()}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return written


def write_csv(df: pd.DataFrame, output_dir: Path, base_name: str) -> Path:
    ensure_dir(output_dir)
    path = output_dir / f"{base_name}.csv"
    df.to_csv(path, index=False)
    return path
