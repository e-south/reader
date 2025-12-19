"""
<reader project>
src/reader/lib/logic_symmetry/overlay.py

Ideal overlay definitions for the logic-symmetry plot.

Adds a tile-rendering mode (4-square horizontal strip for 00,10,01,11) and
a 'tiles_dual' gate set including antagonistic/negative counterparts.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .metrics import logic_asym


@dataclass(frozen=True)
class OverlayStyle:
    # mode: "dot" (legacy) or "tiles" (4-square horizontal strip for 00/10/01/11)
    mode: str = "dot"

    # shared appearance
    alpha: float = 0.25
    face_color: str = "#FFFFFF"
    edge_color: str = "#888888"
    show_labels: bool = True
    label_offset: float = 0.02
    label_line_height: float = 0.018
    label_fontsize: int = 12  # NEW: overlay label font

    # dot-specific
    size: float = 40.0

    # tile-specific
    tile_cell_w: float = 0.035
    tile_cell_h: float = 0.035
    tile_gap: float = 0.0
    tile_edge_width: float = 0.6
    tiles_stack_multiple: bool = True


# Base gate sets (u = [u00,u10,u01,u11])
_GATESETS: Dict[str, Dict[str, List[float]]] = {
    "core": {
        "AND":   [0, 0, 0, 1],
        "OR":    [0, 1, 1, 1],
        "XOR":   [0, 1, 1, 0],
        "SLOPE": [0, 0.5, 0.5, 1.0],
        "SIG-A": [0, 1, 0, 1],
        "SIG-B": [0, 0, 1, 1],
    },
    "logic_family": {
        "AND":   [0, 0, 0, 1],
        "OR":    [0, 1, 1, 1],
        "XOR":   [0, 1, 1, 0],
        "NAND":  [1, 1, 1, 0],
        "NOR":   [1, 0, 0, 0],
        "XNOR":  [1, 0, 0, 1],
        "SIG-A": [0, 1, 0, 1],
        "SIG-B": [0, 0, 1, 1],
        "SLOPE": [0, 0.5, 0.5, 1.0],
    },
    "full16": {
        "FALSE": [0, 0, 0, 0],
        "TRUE":  [1, 1, 1, 1],
        "AND":   [0, 0, 0, 1],
        "OR":    [0, 1, 1, 1],
        "XOR":   [0, 1, 1, 0],
        "XNOR":  [1, 0, 0, 1],
        "NAND":  [1, 1, 1, 0],
        "NOR":   [1, 0, 0, 0],
        "A":     [0, 1, 0, 1],
        "NOT A": [1, 0, 1, 0],
        "B":     [0, 0, 1, 1],
        "NOT B": [1, 1, 0, 0],
        "A AND NOT B": [0, 1, 0, 0],
        "NOT A AND B": [0, 0, 1, 0],
        "A->B":  [1, 1, 0, 1],
        "B->A":  [1, 0, 1, 1],
    },
}


def _invert_u(u: List[float]) -> List[float]:
    return [float(1.0 - float(x)) for x in u]


def _tiles_dual_set() -> Dict[str, List[float]]:
    """OR/NOR, XNOR/AND, SIG-A/NOT A, SIG-B/NOT B, SLOPE/SLOPE− (antagonistic)."""
    base = {
        "OR":    [0, 1, 1, 1],
        "NOR":   [1, 0, 0, 0],
        "XNOR":  [1, 0, 0, 1],
        "AND":   [0, 0, 0, 1],
        "SIG-A": [0, 1, 0, 1],
        "SIG-B": [0, 0, 1, 1],
        "SLOPE": [0, 0.5, 0.5, 1.0],
    }
    dual = {
        "NAND":  [1, 1, 1, 0],
        "XOR":   [0, 1, 1, 0],
        "NOT A": [1, 0, 1, 0],
        "NOT B": [1, 1, 0, 0],
        "SLOPE−": _invert_u(base["SLOPE"]),
    }
    out = dict(base)
    out.update(dual)
    return out


def generate_overlay_points(gate_set: str) -> pd.DataFrame:
    """
    Return DataFrame with columns: label, L, A, u  (u = [u00,u10,u01,u11])
    """
    if gate_set == "tiles_dual":
        src = _tiles_dual_set()
    else:
        if gate_set not in _GATESETS:
            valid = list(_GATESETS) + ["tiles_dual"]
            raise ValueError(f"Unknown gate_set '{gate_set}'. Choose from {valid}")
        src = _GATESETS[gate_set]

    rows: List[Dict[str, object]] = []
    for label, u in src.items():
        uu = [float(x) for x in u]
        L, A = logic_asym(np.array(uu, dtype=float))
        rows.append(dict(label=label, L=L, A=A, u=uu))
    return pd.DataFrame.from_records(rows)
