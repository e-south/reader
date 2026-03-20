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
    bootstrap_mean_interval,
    colors_for,
    emit_plot_figure,
    require_columns,
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
    df = trace[trace["metric"].isin(selected_metrics)].copy()
    if only_control:
        df = df[df["sponge"].astype(str) == str(control_name)]
    elif not include_control:
        df = df[df["sponge"].astype(str) != str(control_name)]
    if relevant_only:
        if "relevant_sensor_pair" not in df.columns:
            raise ValueError("retron_sponge_trace: relevant_only requires relevant_sensor_pair in the trace table.")
        df = df[df["relevant_sensor_pair"].fillna(False)]
    if warn_if_empty(df, where="retron_sponge_trace", detail="after metric/control filters"):
        return []

    sensors = _ordered(df["sensor"].tolist())
    figures: list[PlotFigure] = []
    for sensor in sensors:
        sensor_df = df[df["sensor"].astype(str) == sensor].copy()
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
                                    color=color_map[sponge],
                                    label=f"{sponge} {iptg}",
                                    linestyle="--" if str(iptg) == "+IPTG" else "-",
                                    legend_handles=legend_handles,
                                )
                        else:
                            _plot_trace_line(
                                ax=ax,
                                df=sponge_df,
                                color=color_map[sponge],
                                label=str(sponge),
                                linestyle="-",
                                legend_handles=legend_handles,
                            )
                    if metric in {"B", "C", "D", "M", "O", "L_pre"}:
                        ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
                    ax.set_title(_stress_panel_label(str(stress)), pad=8)
                    if col_idx == 0:
                        ax.set_ylabel(_metric_axis_label(metric, metric_label_map=metric_label_map))
                    if row_idx == rows - 1:
                        ax.set_xlabel("Time from stress addition (h)")
                    with suppress(Exception):
                        ax.set_box_aspect(1.0)
            fig.suptitle(
                f"{title}\nSensor: {sensor}",
                y=float(fig_kwargs.get("suptitle_y", 0.98)),
                x=0.5,
                ha="center",
                fontweight="normal",
                linespacing=1.2,
            )
            fig.supxlabel("Time from stress addition (h)", y=0.06)
            if legend_handles:
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
                top=0.84,
                bottom=0.12,
                left=0.12,
                right=0.80 if legend_handles else 0.98,
                hspace=0.30,
                wspace=0.18,
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
    burden_metric: str = "T_growth_AUC",
    fig_kwargs: dict | None = None,
) -> list[PlotFigure]:
    require_columns(summary, ["sensor", "sponge", "metric", "value"], where="retron_sponge_summary")
    fig_kwargs = fig_kwargs or {}
    if view == "interaction":
        return _plot_retron_interaction_summary(
            summary=summary,
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
    require_columns(summary, ["stress_condition", "IPTG"], where="retron_interaction_summary")
    df = summary[summary["metric"].astype(str) == metric].copy()
    df = df[df["sponge"].astype(str) != str(control_name)]
    if relevant_only:
        _require_relevant_sensor_pair(df, where="retron_interaction_summary")
        df = df[df["relevant_sensor_pair"].fillna(False)]
    if warn_if_empty(df, where="retron_interaction_summary", detail=metric):
        return []
    df["state"] = df.apply(_state_label, axis=1, no_stress_label=no_stress_label)
    states = list(state_order or _ordered(df["state"].tolist()))
    figures: list[PlotFigure] = []
    for sensor in _ordered(df["sensor"].tolist()):
        sensor_df = df[df["sensor"].astype(str) == sensor].copy()
        sponges = _sponge_levels(sensor_df, control_name=control_name)
        palette = colors_for(max(1, len(sponges)), palette_book)
        color_map = {name: palette[idx % len(palette)] for idx, name in enumerate(sponges)}
        with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
            fig, ax = plt.subplots(
                figsize=tuple(fig_kwargs.get("figsize", (8.5, 4.8))),
                constrained_layout=False,
            )
            x_positions = np.arange(len(states), dtype=float)
            width = 0.8 / max(1, len(sponges))
            for idx, sponge in enumerate(sponges):
                sponge_df = sensor_df[sensor_df["sponge"].astype(str) == sponge]
                values = []
                for state in states:
                    row = sponge_df[sponge_df["state"] == state]
                    values.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
                offsets = x_positions - 0.4 + (idx + 0.5) * width
                ax.bar(offsets, values, width=width, color=color_map[sponge], label=sponge)
            ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(states, rotation=15, ha="right")
            ax.set_ylabel(_summary_metric_label(metric))
            ax.set_title(str(sensor), pad=8)
            with suppress(Exception):
                ax.set_box_aspect(1.0)
            ax.legend(
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                ncol=1,
                title=None,
                borderaxespad=0.0,
            )
            fig.suptitle(str(title), y=float(fig_kwargs.get("suptitle_y", 0.98)), fontweight="normal")
            fig.subplots_adjust(bottom=0.18, top=0.86, left=0.12, right=0.78)
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
            f"No stress induced effect ({no_stress_label}; D_AUC)",
        ),
        ("D_AUC", lambda data: data["is_relevant_stress"].fillna(False), "Relevant stress induced effect (D_AUC)"),
        ("M_AUC", lambda data: pd.Series(True, index=data.index), "Stress modulation (M_AUC)"),
        ("S_AUC", lambda data: pd.Series(True, index=data.index), "Scaled on-target effect (S_AUC)"),
    )
    panel_shapes: list[tuple[int, int]] = []
    for metric, predicate, _panel_title in panels:
        metric_df = df[df["metric"].astype(str) == metric].copy()
        metric_df = metric_df[predicate(metric_df)].copy()
        pivot = _pivot_summary(metric_df)
        panel_shapes.append((max(1, len(pivot.index)), max(1, len(pivot.columns))))
    max_rows = max(shape[0] for shape in panel_shapes)
    max_cols = max(shape[1] for shape in panel_shapes)
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=tuple(
                fig_kwargs.get(
                    "figsize",
                    (max(10.5, 4.6 + 0.95 * max_cols), max(6.8, 4.2 + 0.75 * max_rows * 2)),
                )
            ),
            constrained_layout=False,
        )
        for ax, (metric, predicate, panel_title) in zip(axes.ravel(), panels, strict=False):
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
                square=True,
                linewidths=0.4,
                linecolor="#f0f0f0",
                annot_kws={"fontsize": 8},
            )
            ax.set_title(panel_title, pad=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelrotation=45)
            ax.tick_params(axis="y", labelrotation=0)
            for label in ax.get_xticklabels():
                label.set_ha("right")
        fig.suptitle(str(title), y=float(fig_kwargs.get("suptitle_y", 0.98)), fontweight="normal")
        fig.supxlabel("Sponge design (sponge)", y=0.06)
        fig.supylabel("Sensor", x=0.02)
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.10, right=0.98, hspace=0.34, wspace=0.24)
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _plot_retron_stress_modulation(
    *,
    summary: pd.DataFrame,
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
    df = summary[
        (summary["metric"].astype(str) == metric) & (summary["sponge"].astype(str) != str(control_name))
    ].copy()
    if relevant_only:
        _require_relevant_sensor_pair(df, where="retron_stress_modulation")
        df = df[df["relevant_sensor_pair"].fillna(False)]
    if warn_if_empty(df, where="retron_stress_modulation", detail=metric):
        return []
    sensors = _ordered(df["sensor"].tolist())
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        fig, axes = plt.subplots(
            1,
            max(1, len(sensors)),
            figsize=tuple(fig_kwargs.get("figsize", (4.8 * max(1, len(sensors)), 4.8))),
            constrained_layout=False,
            squeeze=False,
        )
        for idx, sensor in enumerate(sensors):
            ax = axes[0][idx]
            sensor_df = df[df["sensor"].astype(str) == sensor].copy()
            sensor_df = sensor_df.sort_values("value", ascending=True)
            sponges = sensor_df["sponge"].astype(str).tolist()
            families = sensor_df["sponge_family_size"].fillna("other").astype(str).tolist()
            family_levels = _ordered(families, preferred=("mono", "bi", "tri", "quad", "control", "other"))
            family_palette = colors_for(max(1, len(family_levels)), palette_book)
            family_colors = {name: family_palette[i % len(family_palette)] for i, name in enumerate(family_levels)}
            ax.barh(sponges, sensor_df["value"], color=[family_colors.get(item, "#4c72b0") for item in families])
            ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle=":")
            ax.set_title(str(sensor), pad=10)
            ax.set_xlabel("")
            if idx == 0:
                ax.set_ylabel("Sponge design (sponge)")
            with suppress(Exception):
                ax.set_box_aspect(1.0)
        fig.suptitle(str(title), y=float(fig_kwargs.get("suptitle_y", 0.98)), fontweight="normal")
        fig.supxlabel(_summary_metric_label(metric), y=0.06)
        fig.subplots_adjust(top=0.86, bottom=0.16, left=0.18, right=0.98, wspace=0.34)
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _plot_retron_pareto(
    *,
    summary: pd.DataFrame,
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
    sensor_pairs = relevant[relevant["metric"].astype(str) == "S_AUC"][["sponge", "sensor"]].drop_duplicates()
    burden_rows = summary[
        (summary["metric"].astype(str) == burden_metric) & (summary["sponge"].astype(str) == str(control_name))
    ][["sensor", "value"]].rename(columns={"value": "burden_value"})
    if burden_rows.empty:
        raise ValueError(f"retron_pareto: burden metric {burden_metric!r} is missing from the summary table")
    burden = (
        sensor_pairs.merge(burden_rows, on="sensor", how="left").groupby("sponge", dropna=False)["burden_value"].mean()
    )
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
        ax.set_ylabel(f"tetO burden summary ({burden_metric})")
        ax.set_title("Candidate efficacy versus burden", pad=8)
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
        fig.suptitle(str(title), y=float(fig_kwargs.get("suptitle_y", 0.98)), fontweight="normal")
        fig.subplots_adjust(top=0.90, right=0.80)
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _plot_trace_line(
    *, ax, df: pd.DataFrame, color: str, label: str, linestyle: str, legend_handles: dict[str, object]
) -> None:
    summary = _trace_summary_frame(df)
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


def _metric_axis_label(metric: str, *, metric_label_map: Mapping[str, str] | None = None) -> str:
    if metric_label_map and str(metric) in metric_label_map:
        return str(metric_label_map[str(metric)])
    labels = {
        "B": "Baseline-shifted ratio (B)",
        "C": "Matched-control-normalized ratio (C)",
        "D": "Induced sponge effect (D)",
        "M": "Stress modulation (M)",
        "O": "Sign-corrected effect (O)",
        "R": "Dual-reporter ratio (R)",
        "mu": "Growth-rate estimate (mu)",
    }
    return labels.get(str(metric), f"Retron sponge metric ({metric})")


def _summary_metric_label(metric: str) -> str:
    labels = {
        "C_AUC": "Matched-control response (C_AUC)",
        "C_END": "Matched-control endpoint (C_END)",
        "D_AUC": "Induced sponge effect (D_AUC)",
        "M_AUC": "Stress modulation (M_AUC)",
        "S_AUC": "Scaled on-target effect (S_AUC)",
    }
    return labels.get(str(metric), f"Retron sponge summary metric ({metric})")


def _stress_panel_label(stress: str) -> str:
    if not stress:
        return "Stress condition not declared"
    return f"Stress condition: {stress}"


def _first_non_null(series: pd.Series) -> str:
    for value in series:
        if pd.notna(value):
            return str(value)
    return "other"


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
