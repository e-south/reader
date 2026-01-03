import pytest

from reader.lib.microplates.style import PaletteBook


def test_palette_book_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        PaletteBook("nope").colors(1)
