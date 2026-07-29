"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/core/test_plot_style.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from reader.plotting.style import PaletteBook


def test_palette_book_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        PaletteBook("nope").colors(1)
