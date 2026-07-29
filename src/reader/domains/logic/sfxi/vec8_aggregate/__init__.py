from __future__ import annotations

from .constants import SFXI_VEC8_RECORD_ID, VEC8_CHANNELS
from .model import LoadedSFXIVec8Source, SFXIVec8Aggregate, SFXIVec8Source
from .reshape import sfxi_vec8_tidy_rows
from .sources import aggregate_sfxi_vec8_sources, load_sfxi_vec8_table


def render_sfxi_vec8_heatmap(*args, **kwargs):
    from .render import render_sfxi_vec8_heatmap as render  # noqa: PLC0415

    return render(*args, **kwargs)


__all__ = [
    "SFXI_VEC8_RECORD_ID",
    "LoadedSFXIVec8Source",
    "SFXIVec8Aggregate",
    "SFXIVec8Source",
    "VEC8_CHANNELS",
    "aggregate_sfxi_vec8_sources",
    "load_sfxi_vec8_table",
    "render_sfxi_vec8_heatmap",
    "sfxi_vec8_tidy_rows",
]
