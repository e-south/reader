from .mpl import ensure_mpl_cache_dir
from .sinks import PlotFigure, normalize_plot_figures, save_plot_figures
from .style import DEFAULT_RC, PaletteBook, available_palettes, new_fig_ax, use_style
from .utils import ensure_dir, save_figure, slugify

__all__ = [
    "DEFAULT_RC",
    "PaletteBook",
    "PlotFigure",
    "available_palettes",
    "ensure_dir",
    "ensure_mpl_cache_dir",
    "new_fig_ax",
    "normalize_plot_figures",
    "save_figure",
    "save_plot_figures",
    "slugify",
    "use_style",
]
