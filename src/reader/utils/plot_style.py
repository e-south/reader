"""
--------------------------------------------------------------------------------
<reader project>
src/reader/utils/plot_style.py

Colour-book utilities
-  “Categorical”  - one colour per distinct label.
-  Automatic reset - every unique key gets its own palette.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence, Dict, Tuple
import seaborn as sns

__all__ = ["PaletteBook"]

class PaletteBook:
    """
    One-stop colour store so every plot uses the same colour for the same
    label within a single subplot/run.

    Parameters
    ----------
    base : str
        Any seaborn palette name (default = "colorblind").

    Usage
    -----
        pal = PaletteBook()
        mapping = pal(labels, key="group1")
    """
    def __init__(self, base: str = "colorblind") -> None:
        self.base = base
        self._cache: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    def __call__(
        self,
        labels: Sequence[str],
        *,
        key: str = "default"
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Return a mapping {label -> rgb-colour}.  If labels are new under `key`,
        assign them distinct colours from the base palette.
        """
        labels = list(labels)
        if key not in self._cache:
            self._cache[key] = {}
        existing = self._cache[key]

        # find labels that need fresh colours
        unknown = [lbl for lbl in labels if lbl not in existing]
        if unknown:
            palette = sns.color_palette(self.base, len(unknown))
            for lbl, col in zip(unknown, palette):
                existing[lbl] = col

        # return colours in the order of `labels`
        return {lbl: existing[lbl] for lbl in labels}
