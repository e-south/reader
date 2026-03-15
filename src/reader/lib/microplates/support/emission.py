from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from reader.core.plot_sinks import PlotFigure

from .filesystem import save_figure


def emit_plot_figure(
    *,
    fig: Any,
    filename: str,
    output_dir: Path | None,
    fig_kwargs: Mapping[str, Any] | None,
) -> list[PlotFigure]:
    ext = str((fig_kwargs or {}).get("ext", "pdf")).lower()
    dpi = (fig_kwargs or {}).get("dpi", None)
    if output_dir is None:
        return [PlotFigure(fig=fig, filename=filename, ext=ext, dpi=dpi)]
    save_figure(fig, Path(output_dir), filename, ext=ext, dpi=dpi)
    plt.close(fig)
    return []
