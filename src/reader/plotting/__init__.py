from __future__ import annotations

import importlib

_EXPORTS = {
    "mpl": {"ensure_mpl_cache_dir"},
    "sinks": {"PlotFigure", "normalize_plot_figures", "save_plot_figures"},
    "style": {"DEFAULT_RC", "PaletteBook", "available_palettes", "new_fig_ax", "use_style"},
    "utils": {"ensure_dir", "save_figure", "slugify"},
}

__all__ = tuple(sorted({name for names in _EXPORTS.values() for name in names}))


def __getattr__(name: str):
    for module_name, names in _EXPORTS.items():
        if name in names:
            module = importlib.import_module(f"reader.plotting.{module_name}")
            return getattr(module, name)
    raise AttributeError(name)
