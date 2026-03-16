"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plotting/utils.py

Shared plotting utilities (filesystem + naming helpers).
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^\w\-.]+", "_", value)
    return re.sub(r"_{2,}", "_", value).strip("_")


def save_figure(fig, output_dir: Path, filename_stub: str, ext: str = "pdf", dpi: int | None = None) -> Path:
    """
    Save figures as PDF by default (print-friendly, vector).
    Use ext="png" if you explicitly need rasters.
    """
    ensure_dir(output_dir)
    out = output_dir / f"{slugify(filename_stub)}.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    return out
