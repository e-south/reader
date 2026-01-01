"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plotting/microplates/style.py

Palette + subplot layout helpers for plotting modules.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt

# Accessible, print-friendly cycles (Okabe–Ito & friends)
_PALETTES: dict[str, list[str]] = {
    "colorblind": [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
        "#F0E442",
        "#000000",
    ],
    "muted": [
        "#4878CF",
        "#6ACC65",
        "#D65F5F",
        "#B47CC7",
        "#C4AD66",
        "#77BEDB",
        "#4D4D4D",
        "#A0A0A0",
    ],
    "tableau": [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
    ],
}


@dataclass(frozen=True)
class PaletteBook:
    """Small helper carried in RunContext (see core/context.py)."""

    name: str = "colorblind"

    def colors(self, n: int) -> list[str]:
        pal = _PALETTES.get(self.name, _PALETTES["colorblind"])
        if n <= len(pal):
            return pal[:n]
        out = []
        while len(out) < n:
            out.extend(pal)
        return out[:n]


_DEFAULT_RC = {
    "figure_figsize": (5, 5),  # overall figure size (user‑tunable)
    "savefig_dpi": 300,  # affects rasterized artists / PNG
    "axes_spines_top": False,
    "axes_spines_right": False,
    "axes_titleweight": "bold",
    "axes_labelweight": "regular",
    "axes_grid": True,
    "grid_alpha": 0.25,
    "grid_linestyle": "-",
    "grid_color": "#B0B0B0",
    "axes_axisbelow": True,  # grid behind bars/lines
    "font_size": 13.0,
    "axes_labelsize": 13.0,
    "axes_titlesize": 14.0,
    "xtick_labelsize": 12.0,
    "ytick_labelsize": 12.0,
    "legend_fontsize": 12.0,
    "legend_title_fontsize": 12.0,
    "xtick_direction": "out",
    "ytick_direction": "out",
    "legend_frameon": False,
    "pdf_fonttype": 42,  # vector text in PDF
    "pdf_compression": 9,  # NEW: 0..9 (9 = smallest files)
    "path_simplify": True,  # can help time‑series
    "path_simplify_threshold": 0.0,
    "agg_path_chunksize": 20000,  # split long paths for vector backends
}


@contextlib.contextmanager
def use_style(rc: dict | None = None, color_cycle: Iterable[str] | None = None):
    """Context manager to push a small, opinionated Matplotlib style."""
    rc = {**_DEFAULT_RC, **(rc or {})}
    # Optional relative scaling for all font sizes (per-plot, via config.yaml → fig.rc.font_scale)
    scale = float(rc.pop("font_scale", 1.0))

    def _s(key: str) -> float:
        v = float(rc.get(key, _DEFAULT_RC[key]))
        return v * scale

    with mpl.rc_context():
        mpl.rcParams.update(
            {
                "figure.figsize": rc["figure_figsize"],
                "savefig.dpi": rc["savefig_dpi"],
                "axes.spines.top": rc["axes_spines_top"],
                "axes.spines.right": rc["axes_spines_right"],
                "axes.titleweight": rc["axes_titleweight"],
                "axes.labelweight": rc["axes_labelweight"],
                "axes.grid": rc["axes_grid"],
                "grid.alpha": rc["grid_alpha"],
                "grid.linestyle": rc["grid_linestyle"],
                "grid.color": rc["grid_color"],
                "axes.axisbelow": rc["axes_axisbelow"],
                "font.size": _s("font_size"),
                "axes.labelsize": _s("axes_labelsize"),
                "axes.titlesize": _s("axes_titlesize"),
                "xtick.labelsize": _s("xtick_labelsize"),
                "ytick.labelsize": _s("ytick_labelsize"),
                "legend.fontsize": _s("legend_fontsize"),
                "legend.title_fontsize": _s("legend_title_fontsize"),
                "xtick.direction": rc["xtick_direction"],
                "ytick.direction": rc["ytick_direction"],
                "legend.frameon": rc["legend_frameon"],
                "pdf.fonttype": rc["pdf_fonttype"],
                "pdf.compression": int(rc["pdf_compression"]),
                "path.simplify": bool(rc["path_simplify"]),
                "path.simplify_threshold": float(rc["path_simplify_threshold"]),
                "agg.path.chunksize": int(rc["agg_path_chunksize"]),
            }
        )
        if color_cycle is not None:
            mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(color_cycle))
        yield


def new_fig_ax(fig_kwargs: dict | None = None):
    """
    Consistent figure construction.
    Only pass kwargs that Matplotlib's Figure/subplots actually understand.
    Plot- or style-level options (e.g. 'cmap', 'ext', 'rc', 'cbar_shrink', custom knobs)
    must be consumed by the caller and must not be forwarded here.
    """
    fkw = dict(fig_kwargs or {})
    # Whitelist: accepted by plt.subplots / Figure in supported Matplotlib versions
    _ALLOWED = {
        "num",
        "figsize",
        "dpi",
        "facecolor",
        "edgecolor",
        "frameon",
        "clear",
        "constrained_layout",
        "layout",
        "squeeze",
        "subplot_kw",
        "gridspec_kw",
        "sharex",
        "sharey",
    }
    fkw = {k: v for k, v in fkw.items() if k in _ALLOWED}
    if "figsize" not in fkw:
        fkw["figsize"] = _DEFAULT_RC["figure_figsize"]
    # Keep layout tight to help legends in upper-right
    fkw.setdefault("constrained_layout", True)
    return plt.subplots(**fkw)
