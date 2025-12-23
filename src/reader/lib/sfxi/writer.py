"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/sfxi/writer.py

Write SFXI outputs (vec8 CSV and a small JSON log).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_outputs(
    *,
    vec8: pd.DataFrame,
    log: dict[str, Any],
    out_dir: Path | str,
    subdir: str,
    vec8_filename: str,
    log_filename: str,
) -> dict[str, Path]:
    out_root = Path(out_dir) / subdir
    _ensure_dir(out_root)

    vec_path = out_root / vec8_filename
    log_path = out_root / log_filename

    vec8.to_csv(vec_path, index=False)
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, sort_keys=True, default=str)

    return {"vec8": vec_path, "log": log_path}
