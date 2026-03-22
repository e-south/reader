from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reader.plotting.sinks import PlotFigure
from reader.plotting.style import use_style

from .. import _retron_sponge_presentation as retron_presentation
from ..common import (
    best_subplot_grid,
    bootstrap_mean_interval,
    emit_plot_figure,
    require_columns,
    shared_numeric_limits,
    warn_if_empty,
)
from .shared import (
    _auc,
    _finalize_summary_figure,
    _new_summary_grid_figure,
    _ordered,
    _RetronSummaryPlotRequest,
    _slug,
    _sponge_levels,
    _SummaryFigurePolicy,
    _SummarySubplotPolicy,
    _wrap_hyphenated_label,
)


@dataclass(frozen=True)
class _InteractionSummaryStatePayload:
    x_position: float
    tick_label: str
    color: str
    mean: float
    lower: float
    upper: float
    point_positions: np.ndarray
    point_values: np.ndarray


@dataclass(frozen=True)
class _InteractionSummaryAxisPayload:
    sponge: str
    states: tuple[_InteractionSummaryStatePayload, ...]
    y_limits: tuple[float, float]
    show_ylabel: bool


@dataclass(frozen=True)
class _InteractionSummaryFigurePayload:
    sensor: str
    rows: int
    cols: int
    axis_payloads: tuple[_InteractionSummaryAxisPayload, ...]


@dataclass(frozen=True)
class _InteractionSummaryAxisPolicy:
    bar_width: float
    bar_edgecolor: str
    bar_linewidth: float
    error_color: str
    error_linewidth: float
    error_capsize: float
    point_size: float
    point_alpha: float
    point_color: str
    zero_line_color: str
    zero_line_linewidth: float
    zero_line_linestyle: str
    tick_size: float
    title_pad: float
    title_fontsize: float
    ylabel_fontsize: float
    box_aspect: float


_INTERACTION_SUMMARY_AXIS_POLICY = _InteractionSummaryAxisPolicy(
    bar_width=0.66,
    bar_edgecolor="black",
    bar_linewidth=0.4,
    error_color="#222222",
    error_linewidth=1.0,
    error_capsize=3.0,
    point_size=22.0,
    point_alpha=0.7,
    point_color="#111111",
    zero_line_color="#777777",
    zero_line_linewidth=1.0,
    zero_line_linestyle=":",
    tick_size=8.0,
    title_pad=6.0,
    title_fontsize=10.0,
    ylabel_fontsize=11.0,
    box_aspect=1.0,
)


def render_interaction_summary_view(request: _RetronSummaryPlotRequest) -> list[PlotFigure]:
    return _plot_retron_interaction_summary(
        summary=request.summary,
        trace=request.trace,
        output_dir=request.output_dir,
        title=request.title,
        filename=request.filename,
        control_name=request.control_name,
        no_stress_label=request.no_stress_label,
        relevant_only=request.relevant_only,
        metric=str(request.metric or "C_AUC"),
        state_order=request.state_order,
        fig_kwargs=request.fig_kwargs,
    )


def _interaction_summary_figure_policy(*, rows: int, cols: int) -> _SummaryFigurePolicy:
    return _SummaryFigurePolicy(
        default_figsize=(4.4 * cols, 3.9 * rows),
        title_y=0.988,
        subtitle_y=0.934,
        xlabel="IPTG and stress state",
        xlabel_y=0.02,
        xlabel_fontsize=11.0,
        adjust=_SummarySubplotPolicy(
            top=0.76,
            bottom=0.24,
            left=0.12,
            right=0.98,
            hspace=0.38,
            wspace=0.24,
        ),
    )


def _plot_retron_interaction_summary(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir,
    title: str,
    filename: str | None,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
    metric: str,
    state_order: Sequence[str] | None,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    del summary
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
    figure_payloads = _interaction_summary_figure_payloads(
        replicate_df=replicate_df,
        control_name=control_name,
        state_keys=state_keys,
        state_label_map=state_label_map,
    )
    figures: list[PlotFigure] = []
    for payload in figure_payloads:
        sensor = payload.sensor
        sensor_trace = trace[trace["sensor"].astype(str) == sensor].copy()
        policy = _interaction_summary_figure_policy(rows=payload.rows, cols=payload.cols)
        with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
            fig, axes = _new_summary_grid_figure(
                rows=payload.rows,
                cols=payload.cols,
                policy=policy,
                fig_kwargs=fig_kwargs,
                sharey=True,
            )
            axes_flat = axes.ravel()
            for axis, axis_payload in zip(axes_flat, payload.axis_payloads, strict=False):
                _plot_interaction_summary_axis(
                    axis,
                    payload=axis_payload,
                    metric=metric,
                )
            for axis in axes_flat[len(payload.axis_payloads) :]:
                axis.set_visible(False)
            _finalize_summary_figure(
                fig,
                policy=policy,
                fig_kwargs=fig_kwargs,
                title=title,
                context=sensor,
                subtitle=retron_presentation.render_summary_text(
                    retron_presentation.summary_metric_text_spec(metric),
                    trace=sensor_trace,
                ),
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


def _interaction_state_palette(state_keys: Sequence[str]) -> dict[str, str]:
    base_palette = {
        "-IPTG/-stress": "#b0b0b0",
        "+IPTG/-stress": "#6f6f6f",
        "-IPTG/+stress": "#56B4E9",
        "+IPTG/+stress": "#0072B2",
    }
    return {str(state_key): base_palette.get(str(state_key), "#4c72b0") for state_key in state_keys}


def _interaction_state_values(sponge_df: pd.DataFrame, *, state_key: str) -> np.ndarray:
    state_df = sponge_df[sponge_df["state_key"] == str(state_key)].copy()
    values = pd.to_numeric(state_df["value"], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _interaction_state_interval(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean, lower, upper = bootstrap_mean_interval(
        values,
        ci=95.0,
        ci_boot=100,
        rng=np.random.default_rng(0),
    )
    return float(mean), float(lower), float(upper)


def _interaction_summary_figure_payloads(
    *,
    replicate_df: pd.DataFrame,
    control_name: str,
    state_keys: Sequence[str],
    state_label_map: Mapping[str, str],
) -> tuple[_InteractionSummaryFigurePayload, ...]:
    state_palette = _interaction_state_palette(state_keys)
    payloads: list[_InteractionSummaryFigurePayload] = []
    for sensor in _ordered(replicate_df["sensor"].tolist()):
        sensor_df = replicate_df[replicate_df["sensor"].astype(str) == sensor].copy()
        sponges = _sponge_levels(sensor_df, control_name=control_name)
        rows, cols = best_subplot_grid(len(sponges))
        y_limits = shared_numeric_limits(
            sensor_df["value"].to_numpy(dtype=float, copy=False),
            center=0.0,
            pad_fraction=0.12,
            min_span=0.10,
        )
        axis_payloads = tuple(
            _interaction_summary_axis_payload(
                sensor_df=sensor_df[sensor_df["sponge"].astype(str) == sponge].copy(),
                sponge=str(sponge),
                state_keys=state_keys,
                state_label_map=state_label_map,
                state_palette=state_palette,
                y_limits=y_limits,
                show_ylabel=axis_index % cols == 0,
            )
            for axis_index, sponge in enumerate(sponges)
        )
        payloads.append(
            _InteractionSummaryFigurePayload(
                sensor=str(sensor),
                rows=rows,
                cols=cols,
                axis_payloads=axis_payloads,
            )
        )
    return tuple(payloads)


def _interaction_summary_axis_payload(
    *,
    sensor_df: pd.DataFrame,
    sponge: str,
    state_keys: Sequence[str],
    state_label_map: Mapping[str, str],
    state_palette: Mapping[str, str],
    y_limits: tuple[float, float],
    show_ylabel: bool,
) -> _InteractionSummaryAxisPayload:
    return _InteractionSummaryAxisPayload(
        sponge=sponge,
        states=tuple(
            _interaction_summary_state_payload(
                sponge_df=sensor_df,
                state_key=str(state_key),
                state_label=state_label_map.get(str(state_key), str(state_key)),
                state_color=state_palette.get(str(state_key), "#4c72b0"),
                x_position=float(index),
            )
            for index, state_key in enumerate(state_keys)
        ),
        y_limits=y_limits,
        show_ylabel=show_ylabel,
    )


def _interaction_summary_state_payload(
    *,
    sponge_df: pd.DataFrame,
    state_key: str,
    state_label: str,
    state_color: str,
    x_position: float,
) -> _InteractionSummaryStatePayload:
    values = _interaction_state_values(sponge_df, state_key=state_key)
    mean, lower, upper = _interaction_state_interval(values)
    if values.size:
        jitter = np.linspace(-0.12, 0.12, num=values.size, dtype=float)
        point_positions = np.full(values.size, x_position, dtype=float) + jitter
    else:
        point_positions = np.array([], dtype=float)
    return _InteractionSummaryStatePayload(
        x_position=x_position,
        tick_label=_format_interaction_state_label(state_label),
        color=state_color,
        mean=mean,
        lower=lower,
        upper=upper,
        point_positions=point_positions,
        point_values=values,
    )


def _plot_interaction_summary_axis(
    ax,
    *,
    payload: _InteractionSummaryAxisPayload,
    metric: str,
) -> None:
    _render_interaction_summary_bars(ax, payload=payload, policy=_INTERACTION_SUMMARY_AXIS_POLICY)
    _render_interaction_summary_points(ax, payload=payload, policy=_INTERACTION_SUMMARY_AXIS_POLICY)
    _decorate_interaction_summary_axis(
        ax,
        payload=payload,
        metric=metric,
        policy=_INTERACTION_SUMMARY_AXIS_POLICY,
    )


def _render_interaction_summary_bars(
    ax,
    *,
    payload: _InteractionSummaryAxisPayload,
    policy: _InteractionSummaryAxisPolicy,
) -> None:
    x_positions = np.array([state.x_position for state in payload.states], dtype=float)
    means = np.array([state.mean for state in payload.states], dtype=float)
    lowers = np.array([state.lower for state in payload.states], dtype=float)
    uppers = np.array([state.upper for state in payload.states], dtype=float)
    ax.bar(
        x_positions,
        means,
        width=policy.bar_width,
        color=[state.color for state in payload.states],
        edgecolor=policy.bar_edgecolor,
        linewidth=policy.bar_linewidth,
        zorder=2,
    )
    if np.isfinite(means).any():
        ax.errorbar(
            x_positions,
            means,
            yerr=np.vstack([means - lowers, uppers - means]),
            fmt="none",
            ecolor=policy.error_color,
            elinewidth=policy.error_linewidth,
            capsize=policy.error_capsize,
            zorder=3,
        )


def _render_interaction_summary_points(
    ax,
    *,
    payload: _InteractionSummaryAxisPayload,
    policy: _InteractionSummaryAxisPolicy,
) -> None:
    for state in payload.states:
        if state.point_values.size == 0:
            continue
        ax.scatter(
            state.point_positions,
            state.point_values,
            s=policy.point_size,
            alpha=policy.point_alpha,
            color=policy.point_color,
            zorder=4,
        )


def _decorate_interaction_summary_axis(
    ax,
    *,
    payload: _InteractionSummaryAxisPayload,
    metric: str,
    policy: _InteractionSummaryAxisPolicy,
) -> None:
    x_positions = np.array([state.x_position for state in payload.states], dtype=float)
    ax.axhline(
        0.0,
        color=policy.zero_line_color,
        linewidth=policy.zero_line_linewidth,
        linestyle=policy.zero_line_linestyle,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([state.tick_label for state in payload.states], rotation=0, ha="center")
    ax.tick_params(axis="both", labelsize=policy.tick_size)
    ax.set_ylim(payload.y_limits)
    ax.set_title(
        _wrap_hyphenated_label(payload.sponge, max_parts_per_line=2),
        pad=policy.title_pad,
        fontweight="normal",
        fontsize=policy.title_fontsize,
    )
    ax.set_ylabel(
        retron_presentation.summary_metric_label(metric) if payload.show_ylabel else "",
        fontsize=policy.ylabel_fontsize,
    )
    with suppress(Exception):
        ax.set_box_aspect(policy.box_aspect)


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
        if "relevant_sensor_pair" not in c_trace.columns:
            raise ValueError("retron_interaction_summary: relevant_sensor_pair is required for on-target filtering")
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
