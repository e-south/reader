from __future__ import annotations

from .constants import VEC8_CHANNELS
from .model import SFXIVec8Aggregate, SFXIVec8Source
from .sources import aggregate_sfxi_vec8_sources


def render_sfxi_vec8_heatmap(*args, **kwargs):
    from .render import render_sfxi_vec8_heatmap as render  # noqa: PLC0415

    return render(*args, **kwargs)


__all__ = [
    "SFXIVec8Aggregate",
    "SFXIVec8Source",
    "VEC8_CHANNELS",
    "aggregate_sfxi_vec8_sources",
    "render_sfxi_vec8_heatmap",
]
