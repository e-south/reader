"""Shared palette + subplot layout helpers for plotting modules."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt

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
    """Small helper carried in RunContext (see workbench/context.py)."""

    name: str = "colorblind"

    def colors(self, n: int) -> list[str]:
        if self.name not in _PALETTES:
            opts = ", ".join(sorted(_PALETTES))
            raise ValueError(f"Unknown palette '{self.name}'. Available: {opts}")
        palette = _PALETTES[self.name]
        if n <= len(palette):
            return palette[:n]
        out = []
        while len(out) < n:
            out.extend(palette)
        return out[:n]


def available_palettes() -> list[str]:
    return sorted(_PALETTES)


DEFAULT_RC = {
    "figure_figsize": (5, 5),
    "savefig_dpi": 300,
    "axes_spines_top": False,
    "axes_spines_right": False,
    "axes_titleweight": "bold",
    "axes_labelweight": "regular",
    "axes_grid": True,
    "grid_alpha": 0.25,
    "grid_linestyle": "-",
    "grid_color": "#B0B0B0",
    "axes_axisbelow": True,
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
    "pdf_fonttype": 42,
    "pdf_compression": 9,
    "path_simplify": True,
    "path_simplify_threshold": 0.0,
    "agg_path_chunksize": 20000,
}


@contextlib.contextmanager
def use_style(rc: dict | None = None, color_cycle: Iterable[str] | None = None):
    """Context manager to push a small, opinionated Matplotlib style."""
    rc = {**DEFAULT_RC, **(rc or {})}
    scale = float(rc.pop("font_scale", 1.0))

    def _scaled(key: str) -> float:
        value = float(rc.get(key, DEFAULT_RC[key]))
        return value * scale

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
                "font.size": _scaled("font_size"),
                "axes.labelsize": _scaled("axes_labelsize"),
                "axes.titlesize": _scaled("axes_titlesize"),
                "xtick.labelsize": _scaled("xtick_labelsize"),
                "ytick.labelsize": _scaled("ytick_labelsize"),
                "legend.fontsize": _scaled("legend_fontsize"),
                "legend.title_fontsize": _scaled("legend_title_fontsize"),
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
    Plot- or style-level options must be consumed by the caller and not forwarded.
    """
    fig_kwargs = dict(fig_kwargs or {})
    allowed = {
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
    fig_kwargs = {key: value for key, value in fig_kwargs.items() if key in allowed}
    if "figsize" not in fig_kwargs:
        fig_kwargs["figsize"] = DEFAULT_RC["figure_figsize"]
    fig_kwargs.setdefault("constrained_layout", True)
    return plt.subplots(**fig_kwargs)


__all__ = ["DEFAULT_RC", "PaletteBook", "available_palettes", "new_fig_ax", "use_style"]
