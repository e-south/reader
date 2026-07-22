"""Compact censor labels for response-window evidence figures."""

from __future__ import annotations

from collections.abc import Mapping

from .sources import STATE_ORDER

_BOUND_GLYPHS = {"exact": "", "lower": "≥", "upper": "≤", "indeterminate": "?"}


def bound_glyph(row: Mapping[str, object], component: str) -> str:
    kind = str(row[f"{component}_bound_kind"])
    try:
        return _BOUND_GLYPHS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported response-window bound kind: {kind!r}.") from exc


def censor_qc_line(row: Mapping[str, object]) -> str:
    components = tuple(f"{prefix}{state}" for prefix in ("r", "b") for state in STATE_ORDER)
    central = sum(bool(bound_glyph(row, component)) for component in components)
    event = sum(
        bool(row[f"{component}_event_sensitivity_has_policy_clipping"])
        or bool(row[f"{component}_event_sensitivity_has_instrument_overflow"])
        for component in components
    )
    return f"Censor QC  central {central}/8 bounded · event envelope {event}/8 affected · ≥ lower · ≤ upper · ? mixed"


def annotate_bound_glyph(axis, *, row: Mapping[str, object], component: str, xy, xytext, ha: str, va: str) -> None:
    glyph = bound_glyph(row, component)
    if not glyph:
        return
    label = axis.annotate(
        glyph,
        xy,
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=8,
        fontweight="semibold",
        color="#374151",
        zorder=6,
    )
    label.set_gid(f"censor-bound-{component}")


__all__ = ["annotate_bound_glyph", "bound_glyph", "censor_qc_line"]
