"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/microplates/support/palette.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from reader.lib.microplates.style import PaletteBook

from .ordering import smart_grouped_dose_key, smart_string_numeric_key


def colors_for(n: int, palette_book: PaletteBook | None) -> list[str]:
    if palette_book:
        if n == 1:
            palette = palette_book.colors(2)
            return [palette[1]] if (palette and str(palette[0]).lower() in {"#000000", "black"}) else [palette[0]]
        return palette_book.colors(n)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        raise RuntimeError("No color cycle available; configure Matplotlib rcParams or provide a PaletteBook.")
    if n == 1 and str(cycle[0]).lower() in {"#000000", "black"} and len(cycle) > 1:
        return [cycle[1]]
    return cycle[:n]


def order_levels(levels: list[str]) -> list[str]:
    try:
        return sorted(levels, key=smart_grouped_dose_key)
    except Exception:
        return sorted(levels, key=smart_string_numeric_key)
