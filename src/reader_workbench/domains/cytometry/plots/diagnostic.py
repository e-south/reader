"""Static diagnostic figure for one resolved cytometry gating workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from reader_workbench.domains.cytometry.analysis import CytometryAnalysisError, prepare_event_table
from reader_workbench.plotting.style import use_style

_DEFINITION_COLUMNS = (
    "cells_enabled",
    "cells_x_channel",
    "cells_x_min",
    "cells_x_max",
    "cells_y_channel",
    "cells_y_min",
    "cells_y_max",
    "singlets_enabled",
    "singlet_x_channel",
    "singlet_y_channel",
    "singlet_ratio_min",
    "singlet_ratio_max",
    "fluorescence_channel",
    "threshold_value",
)


def render_cytometry_diagnostic(
    original_events: pd.DataFrame,
    gate_definition: pd.DataFrame,
    gated_events: pd.DataFrame,
    *,
    max_events: int,
    title: str | None = None,
):
    """Render configured cells, singlets, fluorescence, and retention panels."""

    if max_events <= 0:
        raise CytometryAnalysisError("max_events must be positive.")
    if len(gate_definition) != 1:
        raise CytometryAnalysisError("Cytometry diagnostic requires exactly one resolved gate definition row.")
    missing = [column for column in _DEFINITION_COLUMNS if column not in gate_definition.columns]
    if missing:
        raise CytometryAnalysisError("Cytometry gate definition is missing column(s): " + ", ".join(missing) + ".")
    definition = gate_definition.iloc[0]
    channel_columns = ["fluorescence_channel"]
    if bool(definition["cells_enabled"]):
        channel_columns.extend(("cells_x_channel", "cells_y_channel"))
    if bool(definition["singlets_enabled"]):
        channel_columns.extend(("singlet_x_channel", "singlet_y_channel"))
    channels = tuple(dict.fromkeys(str(definition[column]) for column in channel_columns))
    original_wide = prepare_event_table(pl.from_pandas(original_events), channels=channels).to_pandas()
    _require_wide_columns(gated_events, ("sample_id", "event_index", *channels))
    original_plot = _bounded_events(original_wide, max_events=max_events)
    gated_plot = _bounded_events(gated_events, max_events=max_events)

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.patches import Rectangle  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - dependency guard
        raise CytometryAnalysisError("Cytometry diagnostic plotting requires matplotlib.") from exc

    with use_style({"figure_figsize": (11.0, 8.0), "axes_grid": False}):
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
        cells_ax, singlets_ax, fluorescence_ax, retention_ax = axes.flat
        if bool(definition["cells_enabled"]):
            _scatter_gate(
                cells_ax,
                original_plot,
                gated_plot,
                x_channel=str(definition["cells_x_channel"]),
                y_channel=str(definition["cells_y_channel"]),
            )
            cells_ax.add_patch(
                Rectangle(
                    (float(definition["cells_x_min"]), float(definition["cells_y_min"])),
                    float(definition["cells_x_max"] - definition["cells_x_min"]),
                    float(definition["cells_y_max"] - definition["cells_y_min"]),
                    fill=False,
                    color="#c96845",
                    linewidth=1.6,
                )
            )
        else:
            _mark_gate_disabled(cells_ax)
        cells_ax.set_title("Cells gate")

        if bool(definition["singlets_enabled"]):
            singlet_x = str(definition["singlet_x_channel"])
            singlet_y = str(definition["singlet_y_channel"])
            _scatter_gate(singlets_ax, original_plot, gated_plot, x_channel=singlet_x, y_channel=singlet_y)
            finite_x = pd.to_numeric(original_plot[singlet_x], errors="coerce")
            finite_x = finite_x[np.isfinite(finite_x)]
            if not finite_x.empty:
                line_x = np.array([float(finite_x.min()), float(finite_x.max())])
                for ratio in (float(definition["singlet_ratio_min"]), float(definition["singlet_ratio_max"])):
                    singlets_ax.plot(line_x, ratio * line_x, color="#c96845", linewidth=1.4)
        else:
            _mark_gate_disabled(singlets_ax)
        singlets_ax.set_title("Singlets gate")

        fluor = str(definition["fluorescence_channel"])
        original_fluor = _finite_values(original_plot[fluor])
        gated_fluor = _finite_values(gated_plot[fluor])
        fluorescence_ax.hist(original_fluor, bins=40, color="#b8b8b8", alpha=0.65, label="all")
        fluorescence_ax.hist(gated_fluor, bins=40, color="#315f7d", alpha=0.70, label="gated")
        fluorescence_ax.axvline(float(definition["threshold_value"]), color="#c96845", linewidth=1.6)
        fluorescence_ax.set_xlabel(fluor)
        fluorescence_ax.set_ylabel("Events")
        fluorescence_ax.set_title("Fluorescence")
        fluorescence_ax.legend(frameon=False)

        original_counts = original_wide.groupby("sample_id", sort=True).size()
        gated_counts = gated_events.groupby("sample_id", sort=True).size().reindex(original_counts.index, fill_value=0)
        positions = np.arange(len(original_counts))
        retention_ax.bar(positions - 0.18, original_counts.values, width=0.36, color="#b8b8b8", label="all")
        retention_ax.bar(positions + 0.18, gated_counts.values, width=0.36, color="#315f7d", label="gated")
        retention_ax.set_xticks(positions, original_counts.index.astype(str), rotation=30, ha="right")
        retention_ax.set_ylabel("Events")
        retention_ax.set_title("Final retention")
        retention_ax.legend(frameon=False)
        figure.suptitle(title or "Cytometry gating diagnostic")
    return figure


def _scatter_gate(ax, original: pd.DataFrame, gated: pd.DataFrame, *, x_channel: str, y_channel: str) -> None:
    ax.scatter(original[x_channel], original[y_channel], s=9, alpha=0.18, color="#767676", rasterized=True)
    ax.scatter(gated[x_channel], gated[y_channel], s=10, alpha=0.45, color="#315f7d", rasterized=True)
    ax.set_xlabel(x_channel)
    ax.set_ylabel(y_channel)


def _mark_gate_disabled(ax) -> None:
    ax.text(0.5, 0.5, "Disabled", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _bounded_events(frame: pd.DataFrame, *, max_events: int) -> pd.DataFrame:
    if len(frame) <= max_events:
        return frame.copy()
    return frame.sample(n=max_events, random_state=0).sort_values(["sample_id", "event_index"])


def _finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _require_wide_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise CytometryAnalysisError("Cytometry diagnostic data is missing column(s): " + ", ".join(missing) + ".")
