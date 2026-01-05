"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/plot_utils.py

Shared plotting utilities (filesystem + naming helpers).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slugify(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return re.sub(r"_{2,}", "_", s).strip("_")


def save_figure(fig, output_dir: Path, filename_stub: str, ext: str = "pdf", dpi: int | None = None) -> Path:
    """
    Save figures as **PDF by default** (print-friendly, vector). Use ext="png" if you
    explicitly need rasters.
    """
    ensure_dir(output_dir)
    out = output_dir / f"{slugify(filename_stub)}.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    return out
