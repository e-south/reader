from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reader.plotting.sinks import PlotFigure
from reader.plotting.style import PaletteBook, use_style

from .common import (
    annotate_points_smart,
    best_subplot_grid,
    bootstrap_linear_interval,
    bootstrap_mean_interval,
    colors_for,
    emit_plot_figure,
    require_columns,
    shared_numeric_limits,
    warn_if_empty,
)

_FAMILY_ORDER = {"mono": 0, "bi": 1, "tri": 2, "quad": 3, "control": 4}
_IPTG_ORDER = ("-IPTG", "+IPTG")


def plot_retron_sponge_trace(
    *,
    trace: pd.DataFrame,
    output_dir: Path | None,
    metrics: Sequence[str],
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str = "tetO",
    include_control: bool = False,
    only_control: bool = False,
    relevant_only: bool = False,
    stress_order: Sequence[str] | None = None,
    metric_label_map: Mapping[str, str] | None = None,
    fig_kwargs: dict | None = None,
) -> list[PlotFigure]:
    require_columns(
        trace,
        ["sensor", "sponge", "stress_condition", "time_from_stress", "metric", "value"],
        where="retron_sponge_trace",
    )
    fig_kwargs = fig_kwargs or {}
    selected_metrics = [str(metric) for metric in metrics]
    full_df = trace.copy()
    if only_control:
        full_df = full_df[full_df["sponge"].astype(str) == str(control_name)]
    elif not include_control:
        full_df = full_df[full_df["sponge"].astype(str) != str(control_name)]
    if relevant_only:
        if "relevant_sensor_pair" not in full_df.columns:
            raise ValueError("retron_sponge_trace: relevant_only requires relevant_sensor_pair in the trace table.")
        full_df = full_df[full_df["relevant_sensor_pair"].fillna(False)]
    df = full_df[full_df["metric"].isin(selected_metrics)].copy()
    if warn_if_empty(df, where="retron_sponge_trace", detail="after metric/control filters"):
        return []

    sensors = _ordered(df["sensor"].tolist())
    figures: list[PlotFigure] = []
    for sensor in sensors:
        sensor_df = df[df["sensor"].astype(str) == sensor].copy()
        sensor_full_trace = full_df[full_df["sensor"].astype(str) == sensor].copy()
        stresses = _preferred_stresses(sensor_df["stress_condition"], stress_order=stress_order)
        rows = len(selected_metrics)
        cols = max(1, len(stresses))
        width = float(fig_kwargs.get("figsize", [cols * 4.8, rows * 3.6])[0])
        height = float(fig_kwargs.get("figsize", [cols * 4.8, rows * 3.6])[1])
        with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(width, height),
                constrained_layout=False,
                squeeze=False,
                sharex=True,
                sharey="row",
            )
            legend_handles: dict[str, object] = {}
            for row_idx, metric in enumerate(selected_metrics):
                metric_df = sensor_df[sensor_df["metric"].astype(str) == metric].copy()
                sponge_levels = _sponge_levels(metric_df, control_name=control_name)
                color_values = colors_for(max(1, len(sponge_levels)), palette_book)
                color_map = {name: color_values[idx % len(color_values)] for idx, name in enumerate(sponge_levels)}
                has_iptg = "IPTG" in metric_df.columns and metric_df["IPTG"].notna().any()
                iptg_levels = [value for value in _IPTG_ORDER if value in set(metric_df["IPTG"].dropna().astype(str))]
                if has_iptg and not iptg_levels:
                    iptg_levels = _ordered(metric_df["IPTG"].dropna().tolist())
                for col_idx, stress in enumerate(stresses):
                    ax = axes[row_idx][col_idx]
                    panel = metric_df[metric_df["stress_condition"].astype(str) == stress].copy()
                    if panel.empty:
                        ax.set_visible(False)
                        continue
                    for sponge in sponge_levels:
                        sponge_df = panel[panel["sponge"].astype(str) == sponge]
                        if sponge_df.empty:
                            continue
                        if has_iptg:
                            for iptg in iptg_levels:
                                subgroup = sponge_df[sponge_df["IPTG"].astype(str) == iptg]
                                if subgroup.empty:
                                    continue
                                _plot_trace_line(
                                    ax=ax,
                                    df=subgroup,
                                    full_trace=sensor_full_trace,
                                    metric=metric,
                                    stress_condition=str(stress),
                                    sponge=str(sponge),
                                    color=color_map[sponge],
                                    label=f"{sponge} {iptg}",
                                    linestyle="--" if str(iptg) == "+IPTG" else "-",
                                    legend_handles=legend_handles,
                                )
                        else:
                            _plot_trace_line(
                                ax=ax,
                                df=sponge_df,
                                full_trace=sensor_full_trace,
                                metric=metric,
                                stress_condition=str(stress),
                                sponge=str(sponge),
                                color=color_map[sponge],
                                label=str(sponge),
                                linestyle="-",
                                legend_handles=legend_handles,
                            )
                    if only_control:
                        ax.axhline(0.0, color="#c7c7c7", linewidth=0.9, linestyle="-", alpha=0.95, zorder=0.7)
                    elif metric in {"B", "C", "D", "D_abs", "D_growth", "M", "O", "L_pre"}:
                        ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
                    if metric in {"C", "D", "D_abs", "D_growth", "M", "O"}:
                        _annotate_primary_window(ax, sensor_full_trace, stress_condition=str(stress))
                    ax.axvline(0.0, color="#9e9e9e", linewidth=0.9, linestyle="--", alpha=0.9, zorder=0.8)
                    if row_idx == 0:
                        _set_axis_title(ax, _stress_panel_label(str(stress)), pad=6)
                    else:
                        _set_axis_title(ax, "", pad=6)
                    if col_idx == 0:
                        ax.set_ylabel(_metric_axis_label(metric, metric_label_map=metric_label_map), fontsize=13)
                    with suppress(Exception):
                        ax.set_box_aspect(0.62 if only_control else 1.0)
                if metric_df["value"].notna().any():
                    limit_center = (
                        0.0
                        if metric in {"B", "C", "D", "D_abs", "D_growth", "M", "O", "L_pre"}
                        or (only_control and metric in {"R", "mu"})
                        else None
                    )
                    y_limits = shared_numeric_limits(
                        metric_df["value"].to_numpy(dtype=float, copy=False),
                        center=limit_center,
                        pad_fraction=0.10,
                        min_span=0.05,
                    )
                    for ax in axes[row_idx]:
                        if ax.get_visible():
                            ax.set_ylim(y_limits)
                            if only_control:
                                _annotate_stress_addition(ax)
            _set_figure_header(
                fig,
                title=title,
                context=sensor,
                subtitle=_trace_figure_subtitle(selected_metrics),
                title_y=float(fig_kwargs.get("suptitle_y", 0.988)),
                subtitle_y=float(fig_kwargs.get("subtitle_y", 0.936)),
            )
            fig.supxlabel("Time from stress addition (h)", y=0.02, fontsize=13)
            if legend_handles:
                if only_control:
                    fig.legend(
                        legend_handles.values(),
                        legend_handles.keys(),
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.885),
                        ncol=min(2, len(legend_handles)),
                        frameon=False,
                        title=None,
                        borderaxespad=0.0,
                        columnspacing=1.2,
                        handletextpad=0.5,
                    )
                else:
                    fig.legend(
                        legend_handles.values(),
                        legend_handles.keys(),
                        loc="center left",
                        bbox_to_anchor=(0.82, 0.5),
                        ncol=1,
                        frameon=False,
                        title=None,
                        borderaxespad=0.0,
                    )
            fig.subplots_adjust(
                top=0.81 if only_control else 0.78,
                bottom=0.13 if only_control else 0.16,
                left=0.10 if only_control else 0.12,
                right=0.98 if only_control else (0.80 if legend_handles else 0.98),
                hspace=0.18 if only_control else 0.32,
                wspace=0.02 if only_control else 0.18,
            )
            figures.extend(
                emit_plot_figure(
                    fig=fig,
                    filename=f"{filename or _slug(title)}__sensor={_slug(sensor)}",
                    output_dir=output_dir,
                    fig_kwargs=fig_kwargs,
                )
            )
    return figures


def plot_retron_sponge_summary(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None = None,
    output_dir: Path | None,
    view: str,
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str = "tetO",
    no_stress_label: str = "H2O",
    relevant_only: bool = True,
    metric: str | None = None,
    state_order: Sequence[str] | None = None,
    burden_metric: str = "D_growth_AUC",
    fig_kwargs: dict | None = None,
) -> list[PlotFigure]:
    require_columns(summary, ["sensor", "sponge", "metric", "value"], where="retron_sponge_summary")
    fig_kwargs = fig_kwargs or {}
    if view == "interaction":
        return _plot_retron_interaction_summary(
            summary=summary,
            trace=trace,
            output_dir=output_dir,
            title=title,
            filename=filename,
            palette_book=palette_book,
            control_name=control_name,
            no_stress_label=no_stress_label,
            relevant_only=relevant_only,
            metric=str(metric or "C_AUC"),
            state_order=state_order,
            fig_kwargs=fig_kwargs,
        )
    if view == "heatmap":
        return _plot_retron_library_heatmaps(
            summary=summary,
            trace=trace,
            output_dir=output_dir,
            title=title,
            filename=filename,
            control_name=control_name,
            no_stress_label=no_stress_label,
            relevant_only=relevant_only,
            fig_kwargs=fig_kwargs,
        )
    if view == "stress_modulation":
        return _plot_retron_stress_modulation(
            summary=summary,
            trace=trace,
            output_dir=output_dir,
            title=title,
            filename=filename,
            palette_book=palette_book,
            control_name=control_name,
            no_stress_label=no_stress_label,
            relevant_only=relevant_only,
            metric=str(metric or "M_AUC"),
            fig_kwargs=fig_kwargs,
        )
    if view == "pareto":
        return _plot_retron_pareto(
            summary=summary,
            trace=trace,
            output_dir=output_dir,
            title=title,
            filename=filename,
            palette_book=palette_book,
            control_name=control_name,
            no_stress_label=no_stress_label,
            burden_metric=burden_metric,
            fig_kwargs=fig_kwargs,
        )
    raise ValueError(f"retron_sponge_summary: unsupported view {view!r}")


def _plot_retron_interaction_summary(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir: Path | None,
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
    metric: str,
    state_order: Sequence[str] | None,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    if trace is None:
        raise ValueError("retron_interaction_summary: trace input is required to compute per-state uncertainty")
    require_columns(
        trace,
        ["stress_condition", "IPTG", "replicate_id", "time", "metric", "value"],
        where="retron_interaction_summary",
    )
    replicate_df = _interaction_replicate_summary(
        trace=trace,
        metric=metric,
        control_name=control_name,
        no_stress_label=no_stress_label,
        relevant_only=relevant_only,
    )
    if warn_if_empty(replicate_df, where="retron_interaction_summary", detail=metric):
        return []
    state_keys, state_label_map = _resolve_interaction_states(
        replicate_df=replicate_df,
        no_stress_label=no_stress_label,
        state_order=state_order,
    )
    figures: list[PlotFigure] = []
    state_palette = {
        state_keys[0]: "#b0b0b0",
        state_keys[1]: "#6f6f6f",
        state_keys[2]: "#56B4E9",
        state_keys[3]: "#0072B2",
    }
    for sensor in _ordered(replicate_df["sensor"].tolist()):
        sensor_df = replicate_df[replicate_df["sensor"].astype(str) == sensor].copy()
        sensor_trace = trace[trace["sensor"].astype(str) == sensor].copy()
        sponges = _sponge_levels(sensor_df, control_name=control_name)
        rows, cols = best_subplot_grid(len(sponges))
        y_limits = shared_numeric_limits(
            sensor_df["value"].to_numpy(dtype=float, copy=False),
            center=0.0,
            pad_fraction=0.12,
            min_span=0.10,
        )
        with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=tuple(fig_kwargs.get("figsize", (4.4 * cols, 3.9 * rows))),
                constrained_layout=False,
                squeeze=False,
                sharey=True,
            )
            axes_flat = axes.ravel()
            x_positions = np.arange(len(state_keys), dtype=float)
            for axis, sponge in zip(axes_flat, sponges, strict=False):
                sponge_df = sensor_df[sensor_df["sponge"].astype(str) == sponge].copy()
                summary_rows = []
                for state_key in state_keys:
                    state_df = sponge_df[sponge_df["state_key"] == state_key].copy()
                    values = pd.to_numeric(state_df["value"], errors="coerce").to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        summary_rows.append((np.nan, np.nan, np.nan))
                        continue
                    mean, lower, upper = bootstrap_mean_interval(
                        values,
                        ci=95.0,
                        ci_boot=100,
                        rng=np.random.default_rng(0),
                    )
                    summary_rows.append((mean, lower, upper))
                means = np.asarray([row[0] for row in summary_rows], dtype=float)
                lowers = np.asarray([row[1] for row in summary_rows], dtype=float)
                uppers = np.asarray([row[2] for row in summary_rows], dtype=float)
                axis.bar(
                    x_positions,
                    means,
                    width=0.66,
                    color=[state_palette.get(state_key, "#4c72b0") for state_key in state_keys],
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=2,
                )
                if np.isfinite(means).any():
                    axis.errorbar(
                        x_positions,
                        means,
                        yerr=np.vstack([means - lowers, uppers - means]),
                        fmt="none",
                        ecolor="#222222",
                        elinewidth=1.0,
                        capsize=3.0,
                        zorder=3,
                    )
                for idx, state_key in enumerate(state_keys):
                    state_df = sponge_df[sponge_df["state_key"] == state_key].copy()
                    state_values = pd.to_numeric(state_df["value"], errors="coerce").to_numpy(dtype=float)
                    state_values = state_values[np.isfinite(state_values)]
                    if state_values.size == 0:
                        continue
                    jitter = np.linspace(-0.12, 0.12, num=state_values.size)
                    axis.scatter(
                        np.full(state_values.size, x_positions[idx], dtype=float) + jitter,
                        state_values,
                        s=22,
                        alpha=0.7,
                        color="#111111",
                        zorder=4,
                    )
                axis.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
                axis.set_xticks(x_positions)
                axis.set_xticklabels(
                    [
                        _format_interaction_state_label(state_label_map.get(state_key, state_key))
                        for state_key in state_keys
                    ],
                    rotation=0,
                    ha="center",
                )
                axis.tick_params(axis="both", labelsize=8)
                axis.set_ylim(y_limits)
                axis.set_title(str(sponge), pad=6, fontweight="normal", fontsize=10)
                if axis in axes[:, 0]:
                    axis.set_ylabel(_summary_metric_label(metric), fontsize=11)
                else:
                    axis.set_ylabel("")
                with suppress(Exception):
                    axis.set_box_aspect(1.0)
            for axis in axes_flat[len(sponges) :]:
                axis.set_visible(False)
            _set_figure_header(
                fig,
                title=title,
                context=sensor,
                subtitle=_summary_metric_subtitle(metric, trace=sensor_trace),
                title_y=float(fig_kwargs.get("suptitle_y", 0.988)),
                subtitle_y=float(fig_kwargs.get("subtitle_y", 0.934)),
            )
            fig.supxlabel("IPTG and stress state", y=0.02, fontsize=11)
            fig.subplots_adjust(bottom=0.24, top=0.76, left=0.12, right=0.98, hspace=0.38, wspace=0.24)
            figures.extend(
                emit_plot_figure(
                    fig=fig,
                    filename=f"{filename or _slug(title)}__sensor={_slug(sensor)}",
                    output_dir=output_dir,
                    fig_kwargs=fig_kwargs,
                )
            )
    return figures


def _plot_retron_library_heatmaps(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir: Path | None,
    title: str,
    filename: str | None,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    df = summary[summary["sponge"].astype(str) != str(control_name)].copy()
    if relevant_only:
        _require_relevant_sensor_pair(df, where="retron_library_heatmaps")
        df = df[df["relevant_sensor_pair"].fillna(False)]
    if warn_if_empty(df, where="retron_library_heatmaps", detail="after control/on-target filters"):
        return []
    panels = (
        (
            "D_AUC",
            lambda data: data["stress_condition"].astype(str) == str(no_stress_label),
            f"No-stress D_AUC ({no_stress_label})",
            "auc",
        ),
        (
            "D_AUC",
            lambda data: data["is_relevant_stress"].fillna(False),
            "Sensor-matched stress D_AUC",
            "auc",
        ),
        ("M_AUC", lambda data: pd.Series(True, index=data.index), "Stress-gated change (M_AUC)", "auc"),
        ("S_AUC", lambda data: pd.Series(True, index=data.index), "Scaled on-target score (S_AUC)", "scaled"),
    )
    limit_groups: dict[str, list[float]] = {"auc": [], "scaled": []}
    panel_shapes: list[tuple[int, int]] = []
    for metric, predicate, _panel_title, scale_group in panels:
        metric_df = df[df["metric"].astype(str) == metric].copy()
        metric_df = metric_df[predicate(metric_df)].copy()
        pivot = _pivot_summary(metric_df)
        limit_groups[scale_group].extend(pd.to_numeric(pivot.to_numpy().ravel(), errors="coerce").tolist())
        panel_shapes.append((max(1, len(pivot.index)), max(1, len(pivot.columns))))
    panel_limits = {
        scale_group: shared_numeric_limits(values, center=0.0, pad_fraction=0.02, min_span=0.10)
        for scale_group, values in limit_groups.items()
        if np.isfinite(pd.to_numeric(pd.Series(values), errors="coerce")).any()
    }
    max_rows = max(shape[0] for shape in panel_shapes)
    max_cols = max(shape[1] for shape in panel_shapes)
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=tuple(
                fig_kwargs.get(
                    "figsize",
                    (max(13.0, 2.8 * len(panels) + 0.58 * max_cols * len(panels)), max(3.8, 2.2 + 0.34 * max_rows)),
                )
            ),
            constrained_layout=False,
            squeeze=False,
            sharey=True,
        )
        for panel_index, (ax, (metric, predicate, panel_title, scale_group)) in enumerate(
            zip(axes.ravel(), panels, strict=False)
        ):
            metric_df = df[df["metric"].astype(str) == metric].copy()
            if metric_df.empty:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            metric_df = metric_df[predicate(metric_df)].copy()
            pivot = _pivot_summary(metric_df)
            if pivot.empty:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="vlag",
                center=0.0,
                annot=True,
                fmt=".2f",
                cbar=False,
                square=False,
                linewidths=0.4,
                linecolor="#f0f0f0",
                annot_kws={"fontsize": 8.5},
                vmin=panel_limits.get(scale_group, (None, None))[0],
                vmax=panel_limits.get(scale_group, (None, None))[1],
            )
            _set_axis_title(ax, panel_title, pad=8)
            ax.set_xlabel("")
            ax.set_ylabel("")
            wrapped_columns = [_wrap_hyphenated_label(str(label), max_parts_per_line=2) for label in pivot.columns]
            ax.set_xticklabels(wrapped_columns)
            ax.tick_params(axis="x", labelrotation=0, labelsize=9.0, pad=1)
            for label in ax.get_xticklabels():
                label.set_ha("center")
            if panel_index > 0:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.tick_params(axis="y", labelrotation=0, labelsize=10.0)
        _set_figure_header(
            fig,
            title=title,
            subtitle=_library_heatmap_subtitle(trace=trace),
            title_y=float(fig_kwargs.get("suptitle_y", 0.988)),
            subtitle_y=float(fig_kwargs.get("subtitle_y", 0.940)),
        )
        fig.supxlabel("Sponge", y=0.03, fontsize=13)
        fig.supylabel("Sensor", x=0.02, fontsize=13)
        fig.subplots_adjust(top=0.78, bottom=0.18, left=0.10, right=0.99, hspace=0.12, wspace=0.03)
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _plot_retron_stress_modulation(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir: Path | None,
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
    metric: str,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    plot_df = _stress_modulation_plot_frame(
        summary=summary,
        metric=metric,
        control_name=control_name,
        relevant_only=relevant_only,
    )
    if warn_if_empty(plot_df, where="retron_stress_modulation", detail=metric):
        return []
    sensors = _ordered(plot_df["sensor"].tolist())
    sensor_palette = colors_for(max(1, len(sensors)), palette_book)
    sensor_colors = {sensor: sensor_palette[idx % len(sensor_palette)] for idx, sensor in enumerate(sensors)}
    row_labels = [
        _stress_modulation_row_label(sensor=str(row.sensor), sponge=str(row.sponge))
        for row in plot_df.itertuples(index=False)
    ]
    base_positions = np.arange(len(plot_df), dtype=float)
    bar_height = 0.34
    sample_values = pd.to_numeric(plot_df["sample_value"], errors="coerce").to_numpy(dtype=float)
    control_values = pd.to_numeric(plot_df["control_value"], errors="coerce").to_numpy(dtype=float)
    combined = np.concatenate([sample_values[np.isfinite(sample_values)], control_values[np.isfinite(control_values)]])
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, ax = plt.subplots(
            figsize=tuple(fig_kwargs.get("figsize", (8.9, max(4.8, 1.8 + 0.42 * len(plot_df))))),
            constrained_layout=False,
        )
        x_limits = shared_numeric_limits(
            combined if combined.size else np.array([0.0], dtype=float),
            center=0.0,
            pad_fraction=0.10,
            min_span=0.10,
        )
        edge_colors = np.array(
            [sensor_colors.get(str(sensor), "#4c72b0") for sensor in plot_df["sensor"].astype(str)],
            dtype=object,
        )
        control_mask = np.isfinite(control_values)
        sample_mask = np.isfinite(sample_values)
        if control_mask.any():
            ax.barh(
                base_positions[control_mask] - bar_height / 2.0,
                control_values[control_mask],
                height=bar_height * 0.92,
                color="#f3ebe7",
                edgecolor=edge_colors[control_mask].tolist(),
                linewidth=0.9,
                hatch="//",
                label="tetO reference",
            )
        if sample_mask.any():
            ax.barh(
                base_positions[sample_mask] + bar_height / 2.0,
                sample_values[sample_mask],
                height=bar_height * 0.92,
                color=[sensor_colors.get(str(sensor), "#4c72b0") for sensor in plot_df.loc[sample_mask, "sensor"]],
                edgecolor="#222222",
                linewidth=0.5,
                label="Sample",
            )
        ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle=":")
        ax.set_xlabel(_summary_metric_label(metric), fontsize=11)
        ax.set_ylabel("")
        ax.set_yticks(base_positions)
        ax.set_yticklabels(row_labels)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(x_limits)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.55)
        _set_figure_header(
            fig,
            title=title,
            subtitle=_summary_metric_subtitle(metric, trace=trace),
            title_y=float(fig_kwargs.get("suptitle_y", 0.988)),
            subtitle_y=float(fig_kwargs.get("subtitle_y", 0.938)),
        )
        if control_mask.any() or sample_mask.any():
            ax.legend(
                frameon=False,
                title=None,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0.0,
            )
        fig.subplots_adjust(top=0.82, bottom=0.12, left=0.28, right=0.80)
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _plot_retron_pareto(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir: Path | None,
    title: str,
    filename: str | None,
    palette_book: PaletteBook | None,
    control_name: str,
    no_stress_label: str,
    burden_metric: str,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    relevant = summary[
        (summary["sponge"].astype(str) != str(control_name)) & summary["relevant_sensor_pair"].fillna(False)
    ].copy()
    if warn_if_empty(relevant, where="retron_pareto", detail="after on-target filtering"):
        return []
    score = relevant[relevant["metric"].astype(str) == "S_AUC"].groupby("sponge", dropna=False)["value"].mean()
    leak = relevant[relevant["metric"].astype(str) == "L_pre"].groupby("sponge", dropna=False)["value"].mean()
    family = relevant.groupby("sponge", dropna=False)["sponge_family_size"].agg(_first_non_null)
    burden_rows = summary[
        (summary["metric"].astype(str) == burden_metric)
        & (summary["sponge"].astype(str) != str(control_name))
        & summary["relevant_sensor_pair"].fillna(False)
    ][["sponge", "value"]].rename(columns={"value": "burden_value"})
    if burden_rows.empty:
        raise ValueError(f"retron_pareto: burden metric {burden_metric!r} is missing from the summary table")
    burden = burden_rows.groupby("sponge", dropna=False)["burden_value"].mean()
    table = (
        pd.DataFrame({"on_target": score, "leakiness": leak, "burden": burden, "family": family})
        .reset_index()
        .dropna(subset=["on_target", "burden"])
    )
    if warn_if_empty(table, where="retron_pareto", detail="after aggregation"):
        return []
    family_levels = _ordered(
        table["family"].fillna("other").astype(str).tolist(), preferred=("mono", "bi", "tri", "quad", "other")
    )
    palette = colors_for(max(1, len(family_levels)), palette_book)
    color_map = {name: palette[idx % len(palette)] for idx, name in enumerate(family_levels)}
    sizes = 80.0 + 240.0 * table["leakiness"].abs().fillna(0.0)
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, ax = plt.subplots(figsize=tuple(fig_kwargs.get("figsize", (8.5, 5.5))), constrained_layout=False)
        ax.scatter(
            table["on_target"],
            table["burden"],
            s=sizes,
            c=[color_map.get(str(item), "#4c72b0") for item in table["family"].fillna("other")],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )
        annotate_points_smart(
            ax=ax,
            points=[(float(row["on_target"]), float(row["burden"])) for _, row in table.iterrows()],
            labels=[str(row["sponge"]) for _, row in table.iterrows()],
        )
        ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle=":")
        ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
        ax.set_xlabel("Mean scaled on-target effect across relevant sensors (S_AUC)")
        ax.set_ylabel(_burden_axis_label(burden_metric))
        ax.tick_params(axis="both", labelsize=7)
        with suppress(Exception):
            ax.set_box_aspect(1.0)
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=level, markerfacecolor=color_map[level], markersize=8)
            for level in family_levels
        ]
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                frameon=False,
                title=None,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0.0,
            )
        _set_figure_header(
            fig,
            title=title,
            subtitle=_summary_metric_subtitle("S_AUC", trace=trace),
            title_y=float(fig_kwargs.get("suptitle_y", 0.98)),
            subtitle_y=float(fig_kwargs.get("subtitle_y", 0.942)),
        )
        fig.subplots_adjust(top=0.88, right=0.80)
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _plot_trace_line(
    *,
    ax,
    df: pd.DataFrame,
    full_trace: pd.DataFrame,
    metric: str,
    stress_condition: str,
    sponge: str,
    color: str,
    label: str,
    linestyle: str,
    legend_handles: dict[str, object],
) -> None:
    summary = (
        _derived_trace_summary_frame(
            trace=full_trace,
            metric=metric,
            sponge=sponge,
            stress_condition=stress_condition,
        )
        if metric in {"D", "D_abs", "D_growth", "M", "O"}
        else _trace_summary_frame(df)
    )
    if summary.empty:
        return
    ax.fill_between(
        summary["time_from_stress"],
        summary["lower"],
        summary["upper"],
        alpha=0.16,
        color=color,
        linewidth=0.0,
        zorder=1,
    )
    (line,) = ax.plot(
        summary["time_from_stress"].to_numpy(dtype=float),
        summary["mean"].to_numpy(dtype=float),
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=label,
        zorder=2,
    )
    legend_handles.setdefault(label, line)


def _derived_trace_summary_frame(
    *,
    trace: pd.DataFrame,
    metric: str,
    sponge: str,
    stress_condition: str,
) -> pd.DataFrame:
    source_metric = {"D": "C", "O": "C", "M": "C", "D_abs": "R", "D_growth": "mu"}.get(str(metric), "C")
    base_trace = trace[
        (trace["metric"].astype(str) == source_metric) & (trace["sponge"].astype(str) == str(sponge))
    ].copy()
    if base_trace.empty:
        return pd.DataFrame(columns=["time_from_stress", "mean", "lower", "upper"])
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(0)
    time_groups = base_trace.groupby("time_from_stress", dropna=False, sort=True)
    for time_value, time_group in time_groups:
        if metric in {"D", "O", "D_abs", "D_growth"}:
            stress_group = time_group[time_group["stress_condition"].astype(str) == str(stress_condition)].copy()
            plus = pd.to_numeric(
                stress_group[stress_group["IPTG"].astype(str) == "+IPTG"]["value"],
                errors="coerce",
            ).to_numpy(dtype=float)
            minus = pd.to_numeric(
                stress_group[stress_group["IPTG"].astype(str) == "-IPTG"]["value"],
                errors="coerce",
            ).to_numpy(dtype=float)
            mean, lower, upper = bootstrap_linear_interval(
                [plus, minus],
                coefficients=(1.0, -1.0),
                ci=95.0,
                ci_boot=100,
                rng=rng,
            )
            if metric == "O":
                expected_sign = float(
                    pd.to_numeric(stress_group["expected_decoy_sign"], errors="coerce").dropna().iloc[0]
                )
                mean, lower, upper = expected_sign * mean, expected_sign * lower, expected_sign * upper
        else:
            relevant = time_group[time_group["stress_condition"].astype(str) == str(stress_condition)].copy()
            baseline = time_group[~time_group["is_relevant_stress"].fillna(False)].copy()
            rel_plus = pd.to_numeric(
                relevant[relevant["IPTG"].astype(str) == "+IPTG"]["value"], errors="coerce"
            ).to_numpy(dtype=float)
            rel_minus = pd.to_numeric(
                relevant[relevant["IPTG"].astype(str) == "-IPTG"]["value"], errors="coerce"
            ).to_numpy(dtype=float)
            base_plus = pd.to_numeric(
                baseline[baseline["IPTG"].astype(str) == "+IPTG"]["value"], errors="coerce"
            ).to_numpy(dtype=float)
            base_minus = pd.to_numeric(
                baseline[baseline["IPTG"].astype(str) == "-IPTG"]["value"], errors="coerce"
            ).to_numpy(dtype=float)
            mean, lower, upper = bootstrap_linear_interval(
                [rel_plus, rel_minus, base_plus, base_minus],
                coefficients=(1.0, -1.0, -1.0, 1.0),
                ci=95.0,
                ci_boot=100,
                rng=rng,
            )
        if not np.isfinite(mean):
            continue
        rows.append(
            {
                "time_from_stress": float(time_value),
                "mean": float(mean),
                "lower": float(lower),
                "upper": float(upper),
            }
        )
    return pd.DataFrame(rows).sort_values("time_from_stress", kind="stable").reset_index(drop=True)


def _interaction_replicate_summary(
    *,
    trace: pd.DataFrame,
    metric: str,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
) -> pd.DataFrame:
    c_trace = trace[trace["metric"].astype(str) == "C"].copy()
    c_trace = c_trace[c_trace["sponge"].astype(str) != str(control_name)]
    if relevant_only:
        _require_relevant_sensor_pair(c_trace, where="retron_interaction_summary")
        c_trace = c_trace[c_trace["relevant_sensor_pair"].fillna(False)]
    rows: list[dict[str, object]] = []
    group_columns = [
        "plate_id",
        "sensor",
        "sponge",
        "genotype_id",
        "stress_condition",
        "IPTG",
        "replicate_id",
    ]
    for _, group in c_trace.groupby(group_columns, dropna=False):
        ordered = group.sort_values("time", kind="stable")
        values = pd.to_numeric(ordered["value"], errors="coerce").to_numpy(dtype=float)
        times = pd.to_numeric(ordered["time"], errors="coerce").to_numpy(dtype=float)
        if metric == "C_AUC":
            mask = ordered["in_primary_post_stress"].astype(bool).to_numpy()
            value = _auc(times[mask], values[mask])
        elif metric == "C_END":
            mask = ordered["in_endpoint_window"].astype(bool).to_numpy()
            value = np.nan if not mask.any() else float(np.nanmean(values[mask]))
        else:
            raise ValueError(f"retron_interaction_summary: unsupported metric {metric!r}")
        row = ordered.iloc[0]
        rows.append(
            {
                "plate_id": row["plate_id"],
                "sensor": row["sensor"],
                "sponge": row["sponge"],
                "genotype_id": row["genotype_id"],
                "stress_condition": row["stress_condition"],
                "IPTG": row["IPTG"],
                "replicate_id": row["replicate_id"],
                "state_key": _state_key(row, no_stress_label=no_stress_label),
                "state_label": _state_label(row, no_stress_label=no_stress_label),
                "value": value,
                "expected_decoy_sign": row.get("expected_decoy_sign"),
                "is_relevant_stress": row.get("is_relevant_stress"),
                "relevant_sensor_pair": row.get("relevant_sensor_pair"),
                "sponge_family_size": row.get("sponge_family_size"),
            }
        )
    return pd.DataFrame(rows)


def _trace_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["time_from_stress", "mean", "lower", "upper"])
    grouped = df.groupby("time_from_stress", dropna=False)["value"]
    rng = np.random.default_rng(0)
    rows: list[dict[str, float]] = []
    for time_value, series in grouped:
        mean, lower, upper = bootstrap_mean_interval(
            series.to_numpy(dtype=float, copy=False),
            ci=95.0,
            ci_boot=100,
            rng=rng,
        )
        rows.append(
            {
                "time_from_stress": float(time_value),
                "mean": mean,
                "lower": lower,
                "upper": upper,
            }
        )
    return pd.DataFrame(rows).sort_values("time_from_stress", kind="stable").reset_index(drop=True)


def _pivot_summary(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(index="sensor", columns="sponge", values="value", aggfunc="mean")
    if table.empty:
        return table
    row_order = _ordered(table.index.tolist())
    col_order = sorted(table.columns, key=_sponge_sort_key)
    return table.reindex(index=row_order, columns=col_order)


def _state_label(row: pd.Series, *, no_stress_label: str) -> str:
    iptg = str(row.get("IPTG") or "").strip() or "None"
    stress = str(row.get("stress_condition") or "").strip()
    return f"{stress or no_stress_label} / {iptg}"


def _state_key(row: pd.Series, *, no_stress_label: str) -> str:
    iptg = str(row.get("IPTG") or "").strip() or "-IPTG"
    stress = str(row.get("stress_condition") or "").strip()
    stress_key = "-stress" if not stress or stress == str(no_stress_label) else "+stress"
    return f"{iptg}/{stress_key}"


def _resolve_interaction_states(
    *,
    replicate_df: pd.DataFrame,
    no_stress_label: str,
    state_order: Sequence[str] | None,
) -> tuple[list[str], dict[str, str]]:
    del no_stress_label
    present_rows = (
        replicate_df[["state_key", "state_label"]]
        .drop_duplicates()
        .sort_values(["state_key", "state_label"], kind="stable")
    )
    state_label_map = {str(row["state_key"]): str(row["state_label"]) for _, row in present_rows.iterrows()}
    if state_order:
        ordered_keys = [str(item) for item in state_order if str(item) in state_label_map]
        ordered_keys.extend(key for key in state_label_map if key not in ordered_keys)
        return ordered_keys, state_label_map
    default_order = ("-IPTG/-stress", "+IPTG/-stress", "-IPTG/+stress", "+IPTG/+stress")
    ordered_keys = [key for key in default_order if key in state_label_map]
    ordered_keys.extend(key for key in state_label_map if key not in ordered_keys)
    return ordered_keys, state_label_map


def _format_interaction_state_label(label: str) -> str:
    stress, _, iptg = str(label).partition(" / ")
    if not iptg:
        return str(label)
    return f"{stress}\n{iptg}"


def _metric_axis_label(metric: str, *, metric_label_map: Mapping[str, str] | None = None) -> str:
    if metric_label_map and str(metric) in metric_label_map:
        return str(metric_label_map[str(metric)])
    labels = {
        "B": "Baseline-shifted ratio (B)",
        "C": "Matched-control-normalized ratio (C)",
        "D": "IPTG-state effect (D)",
        "D_abs": "Absolute matched-control effect (D_abs)",
        "D_growth": "Construct-specific growth burden (D_growth)",
        "M": "Stress modulation (M)",
        "O": "Sign-corrected effect (O)",
        "R": "Dual-reporter ratio (R)",
        "mu": "Growth-rate estimate (mu)",
    }
    return labels.get(str(metric), f"Retron sponge metric ({metric})")


def _trace_metric_formula(metric: str) -> str:
    formulas = {
        "R": "R(t)=log2(YFP/CFP)",
        "B": "B(t)=R(t)-R_pre",
        "C": "C(t)=B(t)-B_tetO,matched(t)",
        "D": "D(t)=mean C(+IPTG)-mean C(-IPTG)",
        "D_abs": "D_abs(t)=mean(R-R_tetO,matched)(+IPTG)-mean(R-R_tetO,matched)(-IPTG)",
        "D_growth": "D_growth(t)=mean(mu-mu_tetO,matched)(+IPTG)-mean(mu-mu_tetO,matched)(-IPTG)",
        "M": "M(t)=D(sensor-matched stress)-D(H2O)",
        "O": "O(t)=expected_sign x D(t)",
        "mu": "mu(t)=d ln(OD600) / dt",
    }
    return formulas.get(str(metric), str(metric))


def _trace_figure_subtitle(metrics: Sequence[str]) -> str:
    return "; ".join(_trace_metric_formula(metric) for metric in metrics)


def _summary_metric_label(metric: str) -> str:
    labels = {
        "C_AUC": "Matched-control AUC (C_AUC)",
        "C_END": "Matched-control endpoint (C_END)",
        "D_AUC": "IPTG-state effect AUC (D_AUC)",
        "D_abs_AUC": "Absolute matched-control AUC (D_abs_AUC)",
        "D_growth_AUC": "Construct-specific growth burden AUC (D_growth_AUC)",
        "M_AUC": "Stress modulation AUC (M_AUC)",
        "S_AUC": "Scaled on-target effect (S_AUC)",
    }
    return labels.get(str(metric), f"Retron sponge summary metric ({metric})")


def _summary_metric_formula(metric: str) -> str:
    formulas = {
        "C_AUC": "C_AUC = AUC[C(t)] over the primary post-stress window",
        "C_END": "C_END = mean C(t) over the endpoint window",
        "D_AUC": "D_AUC = AUC[D(t)] over the primary post-stress window",
        "D_abs_AUC": "D_abs_AUC = AUC[D_abs(t)] over the primary post-stress window",
        "D_growth_AUC": "D_growth_AUC = AUC[D_growth(t)] over the primary post-stress window",
        "M_AUC": "M_AUC = AUC[M(t)] over the primary post-stress window",
        "S_AUC": "S_AUC = O_AUC / |G_sensor|",
    }
    return formulas.get(str(metric), str(metric))


def _summary_metric_subtitle(metric: str, *, trace: pd.DataFrame | None = None) -> str:
    base = _summary_metric_formula(metric)
    notes: list[str] = []
    primary_note = (
        _primary_window_span_note_from_trace(trace)
        if str(metric) == "M_AUC"
        else _primary_window_note_from_trace(trace)
    )
    if primary_note:
        notes.append(primary_note)
    if str(metric).endswith("_END"):
        endpoint_note = _endpoint_window_note_from_trace(trace)
        if endpoint_note:
            notes.append(endpoint_note)
    if not notes:
        return base
    return f"{base}; " + "; ".join(notes)


def _annotate_primary_window(ax: plt.Axes, trace: pd.DataFrame, *, stress_condition: str) -> None:
    span = _primary_window_span_bounds(trace, stress_condition=stress_condition)
    if span is None:
        return
    start, end = span
    ax.axvspan(start, end, color="#f3b4b0", alpha=0.14, zorder=0.15, linewidth=0.0)


def _library_heatmap_subtitle(*, trace: pd.DataFrame | None = None) -> str:
    base = "D_AUC = AUC[D(t)]; M_AUC = AUC[M(t)]; S_AUC = O_AUC / |G_sensor|"
    primary_note = _primary_window_note_from_trace(trace)
    if not primary_note:
        return base
    return f"{base}; {primary_note}"


def _primary_window_note_from_trace(trace: pd.DataFrame | None) -> str:
    if trace is None or trace.empty:
        return ""
    required = {"time_from_stress", "in_primary_post_stress"}
    if not required.issubset(trace.columns):
        return ""
    post = trace[trace["in_primary_post_stress"].fillna(False)].copy()
    if post.empty:
        return ""
    notes: list[str] = []
    configured = _configured_primary_window_hours(trace)
    if configured is not None:
        notes.append(f"configured primary window capped at {configured:.1f} h post stress")
    maxima = _window_group_maxima(post)
    if maxima.size == 0:
        return "; ".join(notes)
    notes.append(f"observed primary window reaches {_format_hour_range(maxima)} post stress")
    return "; ".join(notes)


def _primary_window_span_note_from_trace(trace: pd.DataFrame | None) -> str:
    if trace is None or trace.empty:
        return ""
    required = {"time_from_stress", "in_primary_post_stress"}
    if not required.issubset(trace.columns):
        return ""
    post = trace[trace["in_primary_post_stress"].fillna(False)].copy()
    if post.empty:
        return ""
    notes: list[str] = []
    configured = _configured_primary_window_hours(trace)
    if configured is not None:
        notes.append(f"configured primary window capped at {configured:.1f} h post stress")
    minima = _window_group_minima(post)
    maxima = _window_group_maxima(post)
    if minima.size == 0 or maxima.size == 0:
        return "; ".join(notes)
    combined = np.concatenate([minima, maxima])
    notes.append(f"primary post-stress window spans {_format_hour_range(combined)}")
    return "; ".join(notes)


def _configured_primary_window_hours(trace: pd.DataFrame | None) -> float | None:
    if trace is None or trace.empty or "configured_max_post_stress_hours" not in trace.columns:
        return None
    values = pd.to_numeric(trace["configured_max_post_stress_hours"], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return float(values[0])


def _burden_axis_label(metric: str) -> str:
    if str(metric) == "D_growth_AUC":
        return "Mean construct-specific growth burden (D_growth_AUC)"
    if str(metric) == "T_growth_AUC":
        return "Mean tetO growth burden (T_growth_AUC)"
    if str(metric) == "T_finalOD":
        return "Mean tetO endpoint burden (T_finalOD)"
    return f"Burden summary ({metric})"


def _endpoint_window_note_from_trace(trace: pd.DataFrame | None) -> str:
    if trace is None or trace.empty:
        return ""
    required = {"in_endpoint_window", "time"}
    if not required.issubset(trace.columns):
        return ""
    endpoint = trace[trace["in_endpoint_window"].fillna(False)].copy()
    if endpoint.empty:
        return ""
    count = _endpoint_time_count(endpoint)
    if count is None:
        return "endpoint window uses the last flagged reads inside the primary window"
    noun = "read" if count == 1 else "reads"
    return f"endpoint window uses the last {count} flagged {noun} inside that range"


def _window_group_maxima(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["time_from_stress"], errors="coerce")
    group_columns = [column for column in ("plate_id", "sensor", "stress_condition") if column in frame.columns]
    if not group_columns:
        finite = values[np.isfinite(values)]
        return finite.to_numpy(dtype=float, copy=False)
    grouped = frame.assign(__time_from_stress=values).groupby(group_columns, dropna=False)["__time_from_stress"].max()
    finite = pd.to_numeric(grouped, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return finite.to_numpy(dtype=float, copy=False)


def _primary_window_span_bounds(
    trace: pd.DataFrame | None,
    *,
    stress_condition: str,
) -> tuple[float, float] | None:
    if trace is None or trace.empty:
        return None
    required = {"stress_condition", "time_from_stress", "in_primary_post_stress"}
    if not required.issubset(trace.columns):
        return None
    post = trace[
        (trace["stress_condition"].astype(str) == str(stress_condition)) & trace["in_primary_post_stress"].fillna(False)
    ].copy()
    if post.empty:
        return None
    maxima = _window_group_maxima(post)
    finite = maxima[np.isfinite(maxima)]
    if finite.size == 0:
        return None
    end = float(finite.max())
    if end <= 0.0:
        return None
    return 0.0, end


def _window_group_minima(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["time_from_stress"], errors="coerce")
    group_columns = [column for column in ("plate_id", "sensor", "stress_condition") if column in frame.columns]
    if not group_columns:
        finite = values[np.isfinite(values)]
        return finite.to_numpy(dtype=float, copy=False)
    grouped = frame.assign(__time_from_stress=values).groupby(group_columns, dropna=False)["__time_from_stress"].min()
    finite = pd.to_numeric(grouped, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return finite.to_numpy(dtype=float, copy=False)


def _stress_modulation_plot_frame(
    *,
    summary: pd.DataFrame,
    metric: str,
    control_name: str,
    relevant_only: bool,
) -> pd.DataFrame:
    df = summary[summary["metric"].astype(str) == str(metric)].copy()
    sample_df = df[df["sponge"].astype(str) != str(control_name)].copy()
    if relevant_only:
        _require_relevant_sensor_pair(sample_df, where="retron_stress_modulation")
        sample_df = sample_df[sample_df["relevant_sensor_pair"].fillna(False)]
    if sample_df.empty:
        return pd.DataFrame(columns=["sensor", "sponge", "sample_value", "control_value"])
    control_df = df[df["sponge"].astype(str) == str(control_name)].copy()
    join_keys = [
        column
        for column in ("sensor", "stress_condition")
        if column in sample_df.columns
        and column in control_df.columns
        and sample_df[column].notna().any()
        and control_df[column].notna().any()
    ]
    if not join_keys:
        join_keys = ["sensor"]
    control_lookup = control_df.groupby(join_keys, dropna=False)["value"].mean().rename("control_value").reset_index()
    plot_df = sample_df.merge(control_lookup, on=join_keys, how="left")
    plot_df["sample_value"] = pd.to_numeric(plot_df["value"], errors="coerce")
    plot_df["control_value"] = pd.to_numeric(plot_df["control_value"], errors="coerce")
    order = [column for column in ("sensor", "sponge") if column in plot_df.columns]
    if order:
        plot_df = plot_df.sort_values(order, kind="stable")
    keep = [column for column in ("sensor", "sponge", "sample_value", "control_value") if column in plot_df.columns]
    return plot_df[keep].reset_index(drop=True)


def _stress_modulation_row_label(*, sensor: str, sponge: str) -> str:
    return f"{sensor} • {sponge}"


def _endpoint_time_count(frame: pd.DataFrame) -> int | None:
    group_columns = [column for column in ("plate_id", "sensor", "stress_condition") if column in frame.columns]
    if not group_columns:
        count = int(pd.to_numeric(frame["time"], errors="coerce").dropna().nunique())
        return count or None
    counts = (
        frame.assign(__time=pd.to_numeric(frame["time"], errors="coerce"))
        .groupby(group_columns, dropna=False)["__time"]
        .nunique()
    )
    counts = counts[counts > 0]
    if counts.empty:
        return None
    modes = counts.mode(dropna=True)
    if modes.empty:
        return int(counts.iloc[0])
    return int(modes.iloc[0])


def _format_hour_range(values: np.ndarray) -> str:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return ""
    low = float(np.min(finite))
    high = float(np.max(finite))
    if np.isclose(low, high, atol=0.05):
        return f"{high:.1f} h"
    return f"{low:.1f}-{high:.1f} h"


def _stress_panel_label(stress: str) -> str:
    if not stress:
        return "Stress not declared"
    return str(stress)


def _set_axis_title(ax, title: str, *, pad: float = 8.0) -> None:
    ax.set_title(str(title), pad=pad, fontweight="normal", fontsize=10)


def _annotate_stress_addition(ax) -> None:
    if any(text.get_text() == "Stress addition" for text in ax.texts):
        return
    x_limits = ax.get_xlim()
    if len(x_limits) != 2 or not np.isfinite(x_limits).all() or not (x_limits[0] <= 0.0 <= x_limits[1]):
        return
    ax.annotate(
        "Stress addition",
        xy=(0.0, 0.98),
        xycoords=ax.get_xaxis_transform(),
        xytext=(4, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color="#666666",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.25},
        zorder=3.5,
    )


def _set_figure_header(
    fig,
    *,
    title: str,
    context: str | None = None,
    subtitle: str | None = None,
    title_y: float = 0.98,
    subtitle_y: float = 0.945,
) -> None:
    _set_figure_title(fig, title=title, context=context, y=title_y)
    subtitle_text = str(subtitle or "").strip()
    if subtitle_text:
        fig.text(
            0.5,
            subtitle_y,
            subtitle_text,
            ha="center",
            va="top",
            color="#333333",
            fontsize=9,
        )


def _set_figure_title(fig, *, title: str, context: str | None = None, y: float = 0.98) -> None:
    figure_title = str(title).strip()
    context_text = str(context or "").strip()
    if context_text:
        figure_title = f"{figure_title} · {context_text}"
    fig.suptitle(figure_title, y=y, x=0.5, ha="center", fontweight="normal", fontsize=14)


def _wrap_hyphenated_label(label: str, *, max_parts_per_line: int = 2) -> str:
    parts = [part for part in str(label).split("-") if part]
    if len(parts) <= max_parts_per_line:
        return str(label)
    lines = ["-".join(parts[index : index + max_parts_per_line]) for index in range(0, len(parts), max_parts_per_line)]
    return "\n".join(lines)


def _first_non_null(series: pd.Series) -> str:
    for value in series:
        if pd.notna(value):
            return str(value)
    return "other"


def _auc(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) == 0 or len(values) == 0:
        return float("nan")
    finite = np.isfinite(times) & np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.trapezoid(values[finite], times[finite]))


def _require_relevant_sensor_pair(df: pd.DataFrame, *, where: str) -> None:
    if "relevant_sensor_pair" not in df.columns:
        raise ValueError(f"{where}: relevant_sensor_pair is required for on-target filtering")


def _preferred_stresses(values: Iterable[object], *, stress_order: Sequence[str] | None) -> list[str]:
    preferred = [str(value) for value in (stress_order or []) if str(value).strip()]
    return _ordered(values, preferred=preferred or ("H2O",))


def _sponge_levels(df: pd.DataFrame, *, control_name: str) -> list[str]:
    levels = df["sponge"].dropna().astype(str).unique().tolist()
    return sorted(levels, key=lambda item: _sponge_sort_key(item, control_name=control_name))


def _sponge_sort_key(value: str, *, control_name: str = "tetO") -> tuple[int, str]:
    if value == control_name:
        return (_FAMILY_ORDER["control"], value)
    parts = [part for part in str(value).split("-") if part]
    size = {1: "mono", 2: "bi", 3: "tri", 4: "quad"}.get(len(parts), "other")
    return (_FAMILY_ORDER.get(size, 99), str(value))


def _ordered(values: Iterable[object], preferred: Sequence[str] | None = None) -> list[str]:
    seen = {str(value) for value in values if pd.notna(value)}
    ordered = [item for item in (preferred or []) if item in seen]
    ordered.extend(sorted(item for item in seen if item not in ordered))
    return ordered


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value)).strip("_").lower()
