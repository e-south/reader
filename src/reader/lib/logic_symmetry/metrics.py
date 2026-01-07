"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/logic_symmetry/metrics.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS_DEFAULT = 1e-9


@dataclass(frozen=True)
class CornerStats:
    """Aggregated per-corner numbers for a single (design..., batch)."""

    b00: float
    b10: float
    b01: float
    b11: float
    n00: int
    n10: int
    n01: int
    n11: int
    sd00: float
    sd10: float
    sd01: float
    sd11: float


def safe_log(x: np.ndarray | float, eps: float = EPS_DEFAULT) -> np.ndarray | float:
    return np.log(np.maximum(x, eps))


def normalize_u(b00: float, b10: float, b01: float, b11: float, eps: float = EPS_DEFAULT) -> tuple[np.ndarray, float]:
    """
    Returns:
        u: np.array([u00,u10,u01,u11]) in [0,1]
        r: dynamic range (>=1)
    """
    b = np.array([b00, b10, b01, b11], dtype=float)
    b = np.maximum(b, eps)

    m = float(np.min(b))
    M = float(np.max(b))
    r = (M / m) if (M > 0 and m > 0) else 1.0

    if r <= 1.0 + 1e-12:
        # Degenerate range: place all corners mid-scale
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=float), 1.0

    u = (safe_log(b, eps) - safe_log(m, eps)) / (safe_log(r, eps))
    u = np.clip(u, 0.0, 1.0)
    return u, r


def logic_asym(u: np.ndarray) -> tuple[float, float]:
    """
    L = u11 - 0.5*(u10+u01)
    A = u10 - u01
    """
    u00, u10, u01, u11 = [float(x) for x in u]
    L = float(u11 - 0.5 * (u10 + u01))
    A = float(u10 - u01)
    return float(np.clip(L, -1.0, 1.0)), float(np.clip(A, -1.0, 1.0))


def cv_corners(
    sd00: float, sd10: float, sd01: float, sd11: float, b00: float, b10: float, b01: float, b11: float
) -> float:
    """
    Mean CV across corners using per-corner SD / mean, ignoring corners with n<2 or mean<=0.
    """
    pairs = [(sd00, b00), (sd10, b10), (sd01, b01), (sd11, b11)]
    cvs = []
    for sd, mean in pairs:
        if mean > 0:
            cvs.append(float(sd) / float(mean))
    if not cvs:
        return 0.0
    return float(np.mean(cvs))


def compute_metrics(cs: CornerStats, eps: float = EPS_DEFAULT) -> dict[str, float]:
    u, r = normalize_u(cs.b00, cs.b10, cs.b01, cs.b11, eps=eps)
    L, A = logic_asym(u)
    cv = cv_corners(cs.sd00, cs.sd10, cs.sd01, cs.sd11, cs.b00, cs.b10, cs.b01, cs.b11)
    out = {
        "r": float(r),
        "log_r": float(0.0 if r <= 1.0 else safe_log(r)),
        "L": float(L),
        "A": float(A),
        "u00": float(u[0]),
        "u10": float(u[1]),
        "u01": float(u[2]),
        "u11": float(u[3]),
        "cv": float(max(cv, 0.0)),
    }
    return out
