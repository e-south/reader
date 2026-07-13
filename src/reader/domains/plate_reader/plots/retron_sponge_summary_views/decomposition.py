from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis._retron_sponge_contract import DEFAULT_PRIMARY_POST_STRESS_HOURS
from reader.plotting.sinks import PlotFigure
from reader.plotting.style import use_style

from .. import _retron_sponge_presentation as retron_presentation
from .._retron_sponge_trace_support import (
    annotate_primary_window,
    annotate_stress_addition,
    empty_trace_summary_frame,
    grouped_trace_summary_frames,
    trace_display_bounds,
)
from ..common import (
    bootstrap_linear_interval,
    bootstrap_mean_interval,
    emit_plot_figure,
    require_columns,
    shared_numeric_limits,
    warn_if_empty,
)
from .shared import (
    _auc,
    _finalize_summary_figure,
    _ordered,
    _preferred_stresses,
    _require_relevant_sensor_pair,
    _RetronSummaryPlotRequest,
    _slug,
    _SummaryFigurePolicy,
    _SummarySubplotPolicy,
)


@dataclass(frozen=True)
class _DecisionCardRowPayload:
    row_idx: int
    sensor: str
    sponge: str
    stress: str
    panel_limits: tuple[float, float]
    h2o_panel: _DecisionCardPanelPayload
    relevant_panel: _DecisionCardPanelPayload
    summary_strip: _DecisionCardSummaryStripPayload


@dataclass(frozen=True)
class _DecisionCardTraceLinePayload:
    label: str
    color: str
    linestyle: str
    summary: pd.DataFrame


@dataclass(frozen=True)
class _DecisionCardSummaryItemPayload:
    metric: str
    title: str
    label: str
    units: str
    axis_label: str
    point_color: str
    minus_values: tuple[float, ...]
    plus_values: tuple[float, ...]
    contrast_values: tuple[float, ...]
    minus_mean: float | None
    minus_lower: float | None
    minus_upper: float | None
    plus_mean: float | None
    plus_lower: float | None
    plus_upper: float | None
    contrast_mean: float | None
    contrast_lower: float | None
    contrast_upper: float | None
    x_limits: tuple[float, float]


@dataclass(frozen=True)
class _DecisionCardTraceAxisPolicy:
    grid_color: str
    grid_linewidth: float
    grid_alpha: float
    tick_size: float
    ylabel: str
    ylabel_fontsize: float
    legend_loc: str
    legend_fontsize: float
    box_aspect_with_ylabel: float
    box_aspect_without_ylabel: float


@dataclass(frozen=True)
class _DecisionCardSummaryAxisPolicy:
    header_facecolor: str
    header_edgecolor: str
    header_fontsize: float
    header_line_spacing: float
    zero_line_color: str
    zero_line_linewidth: float
    zero_line_linestyle: str
    tick_size: float
    grid_color: str
    grid_linewidth: float
    grid_alpha: float
    label_fontsize: float
    xlabel_fontsize: float
    ylabel_fontsize: float
    ytick_fontsize: float
    box_aspect: float


@dataclass(frozen=True)
class _DecisionCardPanelPayload:
    stress_condition: str
    title: str
    show_ylabel: bool
    show_legend: bool
    panel_trace: pd.DataFrame
    line_payloads: tuple[_DecisionCardTraceLinePayload, ...]


@dataclass(frozen=True)
class _DecisionCardSummaryStripPayload:
    items: tuple[_DecisionCardSummaryItemPayload, ...]


_DECISION_CARD_TRACE_AXIS_POLICY = _DecisionCardTraceAxisPolicy(
    grid_color="#d9d9d9",
    grid_linewidth=0.6,
    grid_alpha=0.45,
    tick_size=7.8,
    ylabel="Reporter ratio R(t)",
    ylabel_fontsize=10.0,
    legend_loc="upper left",
    legend_fontsize=6.8,
    box_aspect_with_ylabel=1.0,
    box_aspect_without_ylabel=1.0,
)

_DECISION_CARD_SUMMARY_AXIS_POLICY = _DecisionCardSummaryAxisPolicy(
    header_facecolor="#f7f7f7",
    header_edgecolor="#d0d0d0",
    header_fontsize=8.1,
    header_line_spacing=1.18,
    zero_line_color="#9e9e9e",
    zero_line_linewidth=1.0,
    zero_line_linestyle=":",
    tick_size=7.8,
    grid_color="#d9d9d9",
    grid_linewidth=0.6,
    grid_alpha=0.45,
    label_fontsize=9.6,
    xlabel_fontsize=8.8,
    ylabel_fontsize=8.8,
    ytick_fontsize=7.8,
    box_aspect=1.0,
)


def render_decomposition_view(request: _RetronSummaryPlotRequest) -> list[PlotFigure]:
    return _plot_retron_decomposition(
        summary=request.summary,
        trace=request.trace,
        output_dir=request.output_dir,
        title=request.title,
        filename=request.filename,
        control_name=request.control_name,
        no_stress_label=request.no_stress_label,
        relevant_only=request.relevant_only,
        fig_kwargs=request.fig_kwargs,
    )


def build_retron_decomposition_frame(
    trace: pd.DataFrame,
    *,
    control_name: str,
    relevant_only: bool,
    summary: pd.DataFrame | None = None,
    no_stress_label: str = "H2O",
) -> pd.DataFrame:
    trace = _validated_decision_card_trace(trace)
    base_frame = _build_primary_window_decomposition_frame(
        trace,
        control_name=control_name,
        relevant_only=relevant_only,
    )
    if summary is None:
        return base_frame
    return _build_decision_card_support_frame(
        trace=trace,
        summary=summary,
        base_frame=base_frame,
        control_name=control_name,
        relevant_only=relevant_only,
        no_stress_label=no_stress_label,
    )


def _build_primary_window_decomposition_frame(
    trace: pd.DataFrame,
    *,
    control_name: str,
    relevant_only: bool,
) -> pd.DataFrame:
    state_auc = _primary_window_auc_frame(trace, metric="R", control_name=control_name, relevant_only=relevant_only)
    if state_auc.empty:
        return _empty_decomposition_frame()
    sample_rows = state_auc[~state_auc["is_control"]].copy()
    control_rows = state_auc[state_auc["is_control"]].copy()
    sample_group = _decomposition_group_columns(sample_rows, include_sponge=True)
    control_group = _decomposition_group_columns(control_rows, include_sponge=False)
    sample_pivot = _pivot_state_auc(sample_rows, index_columns=sample_group, value_prefix="sample")
    control_pivot = _pivot_state_auc(control_rows, index_columns=control_group, value_prefix="control")
    join_columns = [column for column in control_group if column in sample_pivot.columns]
    if join_columns:
        out = sample_pivot.merge(control_pivot, on=join_columns, how="left", validate="many_to_one")
    else:
        out = sample_pivot.assign(
            control_minus_auc=np.nan,
            control_plus_auc=np.nan,
        )
    out["delta_real_auc"] = out["sample_plus_auc"] - out["sample_minus_auc"]
    out["delta_teto_auc"] = out["control_plus_auc"] - out["control_minus_auc"]
    out["delta_net_auc"] = out["delta_real_auc"] - out["delta_teto_auc"]
    order = [column for column in ("sensor", "sponge", "stress_condition", "plate_id") if column in out.columns]
    if order:
        out = out.sort_values(order, kind="stable")
    return out.reset_index(drop=True)


def _empty_decomposition_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sensor",
            "sponge",
            "stress_condition",
            "plate_id",
            "source_experiment_id",
            "source_label",
            "sample_minus_auc",
            "sample_plus_auc",
            "delta_real_auc",
            "control_minus_auc",
            "control_plus_auc",
            "delta_teto_auc",
            "delta_net_auc",
        ]
    )


def _build_decision_card_support_frame(
    *,
    trace: pd.DataFrame,
    summary: pd.DataFrame,
    base_frame: pd.DataFrame,
    control_name: str,
    relevant_only: bool,
    no_stress_label: str,
) -> pd.DataFrame:
    trace_frame = _matched_control_relevant_trace_frame(
        trace,
        metric="R",
        control_name=control_name,
        relevant_only=relevant_only,
        where="retron_decision_card_support",
    )
    if trace_frame.empty:
        return base_frame
    _require_decision_card_summary_columns(summary)
    r_trace = trace[(trace["metric"].astype(str) == "R") & trace["IPTG"].notna()].copy()
    rows: list[dict[str, object]] = []
    for sensor in _ordered(trace_frame["sensor"].astype(str).tolist()):
        sensor_df = trace_frame[trace_frame["sensor"].astype(str) == sensor].copy()
        sample_df = sensor_df[sensor_df["sponge"].astype(str) != str(control_name)].copy()
        sensor_trace = trace[trace["sensor"].astype(str) == sensor].copy()
        sensor_r_trace = r_trace[r_trace["sensor"].astype(str) == sensor].copy()
        for _, spec in _decision_card_row_specs(sample_df).iterrows():
            sponge = str(spec["sponge"])
            stress = str(spec["stress_condition"])
            relevant_sample, relevant_control = _matched_control_condition_frame(
                sensor_r_trace,
                sensor=sensor,
                sponge=sponge,
                stress_condition=stress,
                control_name=control_name,
            )
            h2o_sample, h2o_control = _matched_control_condition_frame(
                sensor_r_trace,
                sensor=sensor,
                sponge=sponge,
                stress_condition=no_stress_label,
                control_name=control_name,
                match_reference=relevant_sample,
            )
            rows.append(
                _decision_card_support_row(
                    sensor_trace=sensor_trace,
                    sensor=sensor,
                    sponge=sponge,
                    stress_condition=stress,
                    primary_stress=stress,
                    panel_role="primary",
                    sample_panel=relevant_sample,
                    control_panel=relevant_control,
                    summary=summary,
                    base_frame=base_frame,
                    control_name=control_name,
                    strict_summary_metrics=True,
                )
            )
            if not h2o_sample.empty or not h2o_control.empty:
                rows.append(
                    _decision_card_support_row(
                        sensor_trace=sensor_trace,
                        sensor=sensor,
                        sponge=sponge,
                        stress_condition=no_stress_label,
                        primary_stress=stress,
                        panel_role="context",
                        sample_panel=h2o_sample,
                        control_panel=h2o_control,
                        summary=summary,
                        base_frame=base_frame,
                        control_name=control_name,
                        strict_summary_metrics=False,
                    )
                )
    if not rows:
        return base_frame
    out = pd.DataFrame(rows)
    order = [
        column
        for column in ("sensor", "sponge", "primary_stress", "panel_role", "stress_condition")
        if column in out.columns
    ]
    if order:
        out = out.sort_values(order, kind="stable")
    return out.reset_index(drop=True)


def _require_decision_card_trace_metadata(trace: pd.DataFrame) -> None:
    require_columns(
        trace,
        [
            "replicate_id",
            "matched_control_key",
            "summary_window_start_h",
            "summary_window_end_h",
            "summary_window_duration_h",
            "in_pre_window",
            "in_primary_post_stress",
            "pre_stress_read_count",
            "post_stress_read_count",
            "matched_group_sample_count",
            "stress_addition_gap_h",
        ],
        where="retron_decomposition",
    )


def _validated_decision_card_trace(trace: pd.DataFrame) -> pd.DataFrame:
    frame = trace.copy()
    require_columns(
        frame,
        ["sensor", "sponge", "stress_condition", "time_from_stress", "metric", "value", "replicate_id"],
        where="retron_decomposition",
    )
    _require_decision_card_trace_metadata(frame)
    return frame


def _require_decision_card_summary_columns(summary: pd.DataFrame) -> None:
    required = {"sensor", "sponge", "stress_condition", "metric", "value"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"retron_decomposition: summary input is missing required columns: {missing}")


def _decision_card_support_row(
    *,
    sensor_trace: pd.DataFrame,
    sensor: str,
    sponge: str,
    stress_condition: str,
    primary_stress: str,
    panel_role: str,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    summary: pd.DataFrame,
    base_frame: pd.DataFrame,
    control_name: str,
    strict_summary_metrics: bool,
) -> dict[str, object]:
    first_row = _panel_reference_row(sample_panel, control_panel)
    base_row = _lookup_primary_window_decomposition_row(
        base_frame,
        sensor=sensor,
        sponge=sponge,
        stress_condition=stress_condition,
    )
    g_sensor_value = _decision_card_sensor_response_value(
        summary=summary,
        sensor=sensor,
        control_name=control_name,
    )
    warning_flag = _decision_card_warning_flag(summary=summary, sensor=sensor, stress=primary_stress)
    record: dict[str, object] = {
        "sensor": sensor,
        "sponge": sponge,
        "stress_condition": stress_condition,
        "primary_stress": primary_stress,
        "panel_role": panel_role,
        "matched_control_key": _panel_metadata_value(
            sample_panel,
            control_panel,
            column="matched_control_key",
        ),
        "sample_minus_auc": _panel_state_auc(sample_panel, iptg="-IPTG"),
        "sample_plus_auc": _panel_state_auc(sample_panel, iptg="+IPTG"),
        "control_minus_auc": _panel_state_auc(control_panel, iptg="-IPTG"),
        "control_plus_auc": _panel_state_auc(control_panel, iptg="+IPTG"),
        "matched_group_sample_count": _panel_metadata_value(
            sample_panel,
            control_panel,
            column="matched_group_sample_count",
            fallback=float(sample_panel["replicate_id"].astype(str).nunique())
            if "replicate_id" in sample_panel.columns
            else np.nan,
        ),
        "pre_stress_read_count": _panel_metadata_value(
            sample_panel,
            control_panel,
            column="pre_stress_read_count",
            fallback=float(_window_read_count(sample_panel, flag_column="in_pre_window") or np.nan),
        ),
        "post_stress_read_count": _panel_metadata_value(
            sample_panel,
            control_panel,
            column="post_stress_read_count",
            fallback=float(_window_read_count(sample_panel, flag_column="in_primary_post_stress") or np.nan),
        ),
        "summary_window_start_h": _panel_metadata_value(sample_panel, control_panel, column="summary_window_start_h"),
        "summary_window_end_h": _panel_metadata_value(sample_panel, control_panel, column="summary_window_end_h"),
        "summary_window_duration_h": _panel_metadata_value(
            sample_panel,
            control_panel,
            column="summary_window_duration_h",
        ),
        "stress_time_zero_h": _panel_metadata_value(sample_panel, control_panel, column="stress_time_zero_h"),
        "stress_addition_gap_h": _panel_metadata_value(sample_panel, control_panel, column="stress_addition_gap_h"),
        "G_sensor": g_sensor_value,
        "warning_flag": warning_flag,
    }
    if first_row is not None:
        for column in ("source_experiment_id", "source_label", "plate_id"):
            if column in first_row.index:
                record[column] = first_row.get(column)
    if base_row is not None:
        for column in (
            "sample_minus_auc",
            "sample_plus_auc",
            "delta_real_auc",
            "control_minus_auc",
            "control_plus_auc",
            "delta_teto_auc",
            "delta_net_auc",
        ):
            if column in base_row:
                record[column] = base_row.get(column)
    record["delta_real_auc"] = _safe_difference(record.get("sample_plus_auc"), record.get("sample_minus_auc"))
    record["delta_teto_auc"] = _safe_difference(record.get("control_plus_auc"), record.get("control_minus_auc"))
    record["delta_net_auc"] = _safe_difference(record.get("delta_real_auc"), record.get("delta_teto_auc"))
    for spec in retron_presentation.decision_card_metric_specs():
        minus_values = np.asarray([], dtype=float)
        plus_values = np.asarray([], dtype=float)
        contrast_values = np.asarray([], dtype=float)
        try:
            minus_series, plus_series = _summary_strip_metric_series(
                metric=spec.metric,
                sensor_trace=sensor_trace,
                sensor=sensor,
                sponge=sponge,
                stress=stress_condition,
                sample_panel=sample_panel,
                control_panel=control_panel,
            )
            minus_values = minus_series.to_numpy(dtype=float, copy=False)
            plus_values = plus_series.to_numpy(dtype=float, copy=False)
            contrast_values = _paired_contrast_values(plus_series=plus_series, minus_series=minus_series)
        except ValueError as exc:
            if "expected_decoy_sign" in str(exc):
                raise
        contrast_mean, contrast_lower, contrast_upper = _summary_strip_contrast_interval(
            plus_values=plus_values,
            minus_values=minus_values,
            contrast_values=contrast_values,
        )
        if contrast_mean is None:
            summary_record = _summary_metric_interval(
                summary,
                sensor=sensor,
                sponge=sponge,
                stress_condition=stress_condition,
                metric=spec.metric,
                strict=strict_summary_metrics,
            )
            if summary_record is None:
                contrast_mean = contrast_lower = contrast_upper = np.nan
            else:
                contrast_mean = summary_record["mean"]
                contrast_lower = summary_record["lower"]
                contrast_upper = summary_record["upper"]
        record[f"{spec.metric}_mean"] = contrast_mean
        record[f"{spec.metric}_lower"] = contrast_lower
        record[f"{spec.metric}_upper"] = contrast_upper
        record[f"{spec.metric}_units"] = spec.units
        record[f"{spec.metric}_minus_values"] = tuple(float(value) for value in minus_values if np.isfinite(value))
        record[f"{spec.metric}_plus_values"] = tuple(float(value) for value in plus_values if np.isfinite(value))
        record[f"{spec.metric}_contrast_values"] = tuple(
            float(value) for value in contrast_values if np.isfinite(value)
        )
        record[f"{spec.metric}_minus_n"] = int(len(minus_values))
        record[f"{spec.metric}_plus_n"] = int(len(plus_values))
        record[f"{spec.metric}_minus_mean"] = _finite_mean(minus_values)
        record[f"{spec.metric}_plus_mean"] = _finite_mean(plus_values)
    burden_record = _summary_metric_interval(
        summary,
        sensor=sensor,
        sponge=sponge,
        stress_condition=stress_condition,
        metric="D_growth_AUC",
        strict=strict_summary_metrics,
    )
    raw_mean = burden_record["mean"] if burden_record is not None else np.nan
    raw_lower = burden_record["lower"] if burden_record is not None else np.nan
    raw_upper = burden_record["upper"] if burden_record is not None else np.nan
    record["D_growth_AUC_mean"] = raw_mean
    record["D_growth_AUC_lower"] = raw_lower
    record["D_growth_AUC_upper"] = raw_upper
    record["D_growth_AUC_units"] = "growth delta x h"
    transformed = _apply_metric_display_transform(
        spec=retron_presentation.DecisionCardMetricSpec(
            metric="D_growth_AUC",
            label="Burden penalty",
            color="#D55E00",
            units="growth delta x h",
            axis_label="AUC[d ln(OD600) / dt]",
            display_multiplier=-1.0,
        ),
        mean=raw_mean,
        lower=raw_lower,
        upper=raw_upper,
    )
    record["D_growth_AUC_display_mean"] = transformed[0]
    record["D_growth_AUC_display_lower"] = transformed[1]
    record["D_growth_AUC_display_upper"] = transformed[2]
    return record


def _panel_reference_row(sample_panel: pd.DataFrame, control_panel: pd.DataFrame) -> pd.Series | None:
    for frame in (sample_panel, control_panel):
        if not frame.empty:
            return frame.iloc[0]
    return None


def _lookup_primary_window_decomposition_row(
    frame: pd.DataFrame,
    *,
    sensor: str,
    sponge: str,
    stress_condition: str,
) -> Mapping[str, object] | None:
    if frame.empty:
        return None
    matches = frame[
        (frame["sensor"].astype(str) == sensor)
        & (frame["sponge"].astype(str) == sponge)
        & (frame["stress_condition"].astype(str) == stress_condition)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _decision_card_support_lookup(
    frame: pd.DataFrame,
    *,
    sensor: str,
    sponge: str,
    primary_stress: str,
    panel_role: str,
) -> Mapping[str, object]:
    if frame.empty:
        raise ValueError("retron_decomposition: support frame is empty for the matched-tetO summary")
    matches = frame[
        (frame["sensor"].astype(str) == sensor)
        & (frame["sponge"].astype(str) == sponge)
        & (frame["primary_stress"].astype(str) == primary_stress)
        & (frame["panel_role"].astype(str) == panel_role)
    ]
    if matches.empty:
        raise ValueError(
            "retron_decomposition: support frame is missing the matched-tetO summary row for "
            f"sensor={sensor!r}, sponge={sponge!r}, stress={primary_stress!r}, panel_role={panel_role!r}"
        )
    return matches.iloc[0]


def _summary_metric_interval(
    summary: pd.DataFrame,
    *,
    sensor: str,
    sponge: str,
    stress_condition: str,
    metric: str,
    strict: bool,
) -> dict[str, float] | None:
    metric_df = summary[
        (summary["sensor"].astype(str) == sensor)
        & (summary["sponge"].astype(str) == sponge)
        & (summary["stress_condition"].astype(str) == stress_condition)
        & (summary["metric"].astype(str) == str(metric))
    ].copy()
    values = pd.to_numeric(metric_df["value"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        if strict:
            raise ValueError(
                "retron_decomposition: summary input is missing required matched-tetO summary metrics for "
                f"sensor={sensor!r}, sponge={sponge!r}, stress={stress_condition!r}: {[str(metric)]!r}"
            )
        return None
    if values.size == 1:
        mean = lower = upper = float(values[0])
    else:
        mean, lower, upper = bootstrap_mean_interval(
            values,
            ci=95.0,
            ci_boot=100,
            rng=np.random.default_rng(0),
        )
    return {"mean": mean, "lower": lower, "upper": upper}


def _panel_state_auc(frame: pd.DataFrame, *, iptg: str) -> float:
    if frame.empty:
        return float("nan")
    state = frame[frame["IPTG"].astype(str) == str(iptg)].copy()
    if state.empty:
        return float("nan")
    ordered = state.sort_values("time_from_stress", kind="stable")
    times = pd.to_numeric(ordered["time_from_stress"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(ordered["value"], errors="coerce").to_numpy(dtype=float)
    return _auc(times, values)


def _panel_metadata_value(
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    *,
    column: str,
    fallback: float | str | None = None,
) -> float | str | None:
    for frame in (sample_panel, control_panel):
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        value = values.iloc[0]
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if np.isfinite(numeric):
            return float(numeric)
        text = str(value).strip()
        if text:
            return text
    return fallback


def _decision_card_warning_flag(
    *,
    summary: pd.DataFrame,
    sensor: str,
    stress: str,
) -> str | None:
    if "warning_flag" not in summary.columns:
        return None
    flags = summary[
        (summary["sensor"].astype(str) == sensor)
        & (summary["stress_condition"].astype(str) == stress)
        & summary["warning_flag"].notna()
    ]["warning_flag"].astype(str)
    if flags.empty:
        return None
    unique = sorted({flag for flag in flags if flag.strip()})
    return ", ".join(unique) if unique else None


def _apply_metric_display_transform(
    *,
    spec: retron_presentation.DecisionCardMetricSpec,
    mean: float,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    values = np.asarray([mean, lower, upper], dtype=float)
    if not np.isfinite(values).any():
        return (np.nan, np.nan, np.nan)
    transformed = values * float(spec.display_multiplier)
    low = float(np.nanmin(transformed))
    high = float(np.nanmax(transformed))
    return float(transformed[0]), low, high


def _safe_difference(left: object, right: object) -> float:
    left_value = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_value = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if np.isfinite(left_value) and np.isfinite(right_value):
        return float(left_value - right_value)
    return float("nan")


def _decomposition_figure_policy(*, row_count: int) -> _SummaryFigurePolicy:
    return _SummaryFigurePolicy(
        default_figsize=(13.8, max(5.1, 3.70 * row_count)),
        title_y=0.982,
        subtitle_y=0.956,
        adjust=_SummarySubplotPolicy(
            top=0.905,
            bottom=0.09,
            left=0.05,
            right=0.96,
            hspace=0.30,
            wspace=0.10,
        ),
    )


def _plot_retron_decomposition(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    output_dir,
    title: str,
    filename: str | None,
    control_name: str,
    no_stress_label: str,
    relevant_only: bool,
    fig_kwargs: dict,
) -> list[PlotFigure]:
    if trace is None:
        raise ValueError("retron_decomposition: trace input is required")
    trace = _validated_decision_card_trace(trace)
    trace_frame = _matched_control_relevant_trace_frame(
        trace,
        metric="R",
        control_name=control_name,
        relevant_only=relevant_only,
        where="retron_decomposition",
    )
    if warn_if_empty(trace_frame, where="retron_decomposition", detail="matched-control R traces"):
        return []
    r_trace = trace[(trace["metric"].astype(str) == "R") & trace["IPTG"].notna()].copy()
    support_frame = build_retron_decomposition_frame(
        trace,
        control_name=control_name,
        relevant_only=relevant_only,
        summary=summary,
        no_stress_label=no_stress_label,
    )
    row_payloads: list[_DecisionCardRowPayload] = []
    for sensor in _ordered(trace_frame["sensor"].astype(str).tolist()):
        sensor_df = trace_frame[trace_frame["sensor"].astype(str) == sensor].copy()
        sensor_r_trace = r_trace[r_trace["sensor"].astype(str) == sensor].copy()
        for _, spec in _decision_card_row_specs(
            sensor_df[sensor_df["sponge"].astype(str) != str(control_name)]
        ).iterrows():
            row_payloads.append(
                _decision_card_row_payload(
                    row_idx=len(row_payloads),
                    sensor_r_trace=sensor_r_trace,
                    sensor=str(sensor),
                    sponge=str(spec["sponge"]),
                    stress=str(spec["stress_condition"]),
                    support_frame=support_frame,
                    control_name=control_name,
                    no_stress_label=no_stress_label,
                )
            )
    if not row_payloads:
        return []
    row_payloads = _apply_shared_summary_x_limits(row_payloads)
    policy = _decomposition_figure_policy(row_count=len(row_payloads))
    display_post_stress_hours = pd.to_numeric(
        pd.Series([fig_kwargs.get("display_post_stress_hours")]),
        errors="coerce",
    ).iloc[0]
    if not np.isfinite(display_post_stress_hours) or display_post_stress_hours <= 0.0:
        span = retron_presentation.primary_window_span_bounds(trace, stress_condition=None)
        display_post_stress_hours = (
            float(span[1])
            if span is not None and np.isfinite(span[1]) and span[1] > 0.0
            else float(DEFAULT_PRIMARY_POST_STRESS_HOURS)
        )
    display_bounds = trace_display_bounds(
        trace,
        max_post_stress_hours=float(display_post_stress_hours),
    )
    with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
        figsize = fig_kwargs.get("figsize", policy.default_figsize)
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        outer = fig.add_gridspec(
            len(row_payloads),
            1,
            left=policy.adjust.left,
            right=policy.adjust.right,
            top=policy.adjust.top,
            bottom=policy.adjust.bottom,
            hspace=policy.adjust.hspace,
        )
        for row_payload in row_payloads:
            _render_decision_card_row(
                figure=fig,
                row_spec=outer[row_payload.row_idx],
                row=row_payload,
                display_bounds=display_bounds,
                show_x_axis_labels=True,
            )
        _finalize_summary_figure(
            fig,
            policy=policy,
            fig_kwargs=fig_kwargs,
            title=title,
            context=None,
            subtitle=retron_presentation.render_summary_text(
                retron_presentation.DECOMPOSITION_TEXT_SPEC,
                trace=trace,
            ),
        )
        return emit_plot_figure(
            fig=fig,
            filename=filename or _slug(title),
            output_dir=output_dir,
            fig_kwargs=fig_kwargs,
        )


def _primary_window_auc_frame(
    trace: pd.DataFrame,
    *,
    metric: str,
    control_name: str,
    relevant_only: bool,
) -> pd.DataFrame:
    require_columns(
        trace,
        [
            "sensor",
            "sponge",
            "stress_condition",
            "IPTG",
            "replicate_id",
            "time_from_stress",
            "metric",
            "value",
            "in_primary_post_stress",
        ],
        where="retron_primary_window_auc_frame",
    )
    frame = trace[trace["metric"].astype(str) == str(metric)].copy()
    frame = frame[frame["in_primary_post_stress"].fillna(False)]
    frame = frame[frame["IPTG"].notna()]
    if frame.empty:
        return pd.DataFrame()
    if relevant_only:
        frame = _matched_control_relevant_trace_frame(
            frame,
            metric=str(metric),
            control_name=control_name,
            relevant_only=True,
            where="retron_primary_window_auc_frame",
        )
        if frame.empty:
            return pd.DataFrame()
    group_columns = [
        column
        for column in (
            "source_experiment_id",
            "source_label",
            "plate_id",
            "sensor",
            "sponge",
            "stress_condition",
            "replicate_id",
            "IPTG",
            "configured_max_post_stress_hours",
        )
        if column in frame.columns
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        record = dict(zip(group_columns, keys, strict=False))
        ordered = group.sort_values("time_from_stress", kind="stable")
        times = pd.to_numeric(ordered["time_from_stress"], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(ordered["value"], errors="coerce").to_numpy(dtype=float)
        record["primary_window_auc"] = _auc(times, values)
        record["is_control"] = str(record.get("sponge", "")) == str(control_name)
        rows.append(record)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sensor"] = out["sensor"].astype(str)
    out["sponge"] = out["sponge"].astype(str)
    out["stress_condition"] = out["stress_condition"].astype(str)
    out["IPTG"] = out["IPTG"].astype(str)
    return out


def _decomposition_group_columns(frame: pd.DataFrame, *, include_sponge: bool) -> list[str]:
    columns = [
        column
        for column in ("source_experiment_id", "source_label", "plate_id", "sensor", "stress_condition")
        if column in frame.columns
    ]
    if include_sponge and "sponge" in frame.columns:
        columns.append("sponge")
    return columns


def _pivot_state_auc(frame: pd.DataFrame, *, index_columns: Sequence[str], value_prefix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*index_columns, f"{value_prefix}_minus_auc", f"{value_prefix}_plus_auc"])
    pivot = (
        frame.pivot_table(
            index=list(index_columns),
            columns="IPTG",
            values="primary_window_auc",
            aggfunc="mean",
        )
        .rename(columns={"-IPTG": f"{value_prefix}_minus_auc", "+IPTG": f"{value_prefix}_plus_auc"})
        .reset_index()
    )
    for expected in (f"{value_prefix}_minus_auc", f"{value_prefix}_plus_auc"):
        if expected not in pivot.columns:
            pivot[expected] = np.nan
    return pivot


def _matched_control_relevant_trace_frame(
    trace: pd.DataFrame,
    *,
    metric: str,
    control_name: str,
    relevant_only: bool,
    where: str,
) -> pd.DataFrame:
    require_columns(
        trace,
        ["sensor", "sponge", "stress_condition", "IPTG", "metric", "value"],
        where=where,
    )
    frame = trace[trace["metric"].astype(str) == str(metric)].copy()
    frame = frame[frame["IPTG"].notna()]
    if frame.empty or not relevant_only:
        return frame
    _require_relevant_sensor_pair(frame, where=where)
    sample_mask = frame["sponge"].astype(str) != str(control_name)
    sample_frame = frame[sample_mask].copy()
    sample_frame = sample_frame[sample_frame["relevant_sensor_pair"].fillna(False)]
    if "is_relevant_stress" in sample_frame.columns:
        sample_frame = sample_frame[sample_frame["is_relevant_stress"].fillna(False)]
    control_frame = frame[~sample_mask].copy()
    match_columns = [
        column
        for column in ("source_experiment_id", "source_label", "plate_id", "sensor", "stress_condition")
        if column in sample_frame.columns and column in control_frame.columns
    ]
    if match_columns and not sample_frame.empty:
        control_frame = control_frame.merge(
            sample_frame[match_columns].drop_duplicates(),
            on=match_columns,
            how="inner",
        )
    elif "is_relevant_stress" in control_frame.columns:
        control_frame = control_frame[control_frame["is_relevant_stress"].fillna(False)]
    return pd.concat([sample_frame, control_frame], ignore_index=True)


def _matched_control_condition_frame(
    trace: pd.DataFrame,
    *,
    sensor: str,
    sponge: str,
    stress_condition: str,
    control_name: str,
    match_reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = trace[
        (trace["sensor"].astype(str) == str(sensor))
        & (trace["stress_condition"].astype(str) == str(stress_condition))
        & trace["IPTG"].notna()
    ].copy()
    sample = frame[frame["sponge"].astype(str) == str(sponge)].copy()
    control = frame[frame["sponge"].astype(str) == str(control_name)].copy()
    match_columns = [
        column
        for column in ("source_experiment_id", "source_label", "plate_id")
        if column in sample.columns and column in control.columns
    ]
    if match_columns:
        reference = match_reference if match_reference is not None else sample
        keys = reference[match_columns].drop_duplicates()
        if not keys.empty:
            sample = sample.merge(keys, on=match_columns, how="inner")
            control = control.merge(keys, on=match_columns, how="inner")
    return sample, control


def _window_read_count(trace: pd.DataFrame, *, flag_column: str) -> int | None:
    if trace.empty or flag_column not in trace.columns or "replicate_id" not in trace.columns:
        return None
    flagged = trace[trace[flag_column].fillna(False)].copy()
    if flagged.empty:
        return None
    counts = (
        flagged.groupby("replicate_id", dropna=False)["time_from_stress"].nunique().to_numpy(dtype=float, copy=False)
    )
    finite = counts[np.isfinite(counts)]
    if finite.size == 0:
        return None
    return int(round(float(np.median(finite))))


def _decision_card_row_specs(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame(columns=["sponge", "stress_condition"])
    stress_levels = _preferred_stresses(
        sample_df["stress_condition"].astype(str).tolist(),
        stress_order=_ordered(sample_df["stress_condition"].astype(str).tolist()),
    )
    out = sample_df.assign(
        __stress_rank=pd.Categorical(
            sample_df["stress_condition"].astype(str),
            categories=stress_levels,
            ordered=True,
        )
    )
    return (
        out[["sponge", "stress_condition", "__stress_rank"]]
        .drop_duplicates()
        .sort_values(["__stress_rank", "sponge"], kind="stable")
        .reset_index(drop=True)
    )


def _decision_card_panel_limits(*frames: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat(
        [pd.to_numeric(frame["value"], errors="coerce") for frame in frames],
        ignore_index=True,
    ).to_numpy(dtype=float, copy=False)
    return shared_numeric_limits(values, center=None, pad_fraction=0.10, min_span=0.12)


def _decision_card_row_payloads(
    *,
    sensor_df: pd.DataFrame,
    sensor_r_trace: pd.DataFrame,
    support_frame: pd.DataFrame,
    control_name: str,
    no_stress_label: str,
) -> list[_DecisionCardRowPayload]:
    sample_df = sensor_df[sensor_df["sponge"].astype(str) != str(control_name)].copy()
    row_specs = _decision_card_row_specs(sample_df)
    return [
        _decision_card_row_payload(
            row_idx=row_idx,
            sensor_r_trace=sensor_r_trace,
            sensor=str(sensor_df["sensor"].astype(str).iloc[0]),
            sponge=str(spec["sponge"]),
            stress=str(spec["stress_condition"]),
            support_frame=support_frame,
            control_name=control_name,
            no_stress_label=no_stress_label,
        )
        for row_idx, spec in row_specs.iterrows()
    ]


def _decision_card_row_payload(
    *,
    row_idx: int,
    sensor_r_trace: pd.DataFrame,
    sensor: str,
    sponge: str,
    stress: str,
    support_frame: pd.DataFrame,
    control_name: str,
    no_stress_label: str,
) -> _DecisionCardRowPayload:
    relevant_sample, relevant_control = _matched_control_condition_frame(
        sensor_r_trace,
        sensor=sensor,
        sponge=sponge,
        stress_condition=stress,
        control_name=control_name,
    )
    h2o_sample, h2o_control = _matched_control_condition_frame(
        sensor_r_trace,
        sensor=sensor,
        sponge=sponge,
        stress_condition=no_stress_label,
        control_name=control_name,
        match_reference=relevant_sample,
    )
    panel_limits = _decision_card_panel_limits(relevant_sample, relevant_control, h2o_sample, h2o_control)
    support_row = _decision_card_support_lookup(
        support_frame,
        sensor=sensor,
        sponge=sponge,
        primary_stress=stress,
        panel_role="primary",
    )
    return _DecisionCardRowPayload(
        row_idx=row_idx,
        sensor=sensor,
        sponge=sponge,
        stress=stress,
        panel_limits=panel_limits,
        h2o_panel=_decision_card_panel_payload(
            row_idx=row_idx,
            sample_panel=h2o_sample,
            control_panel=h2o_control,
            sensor=sensor,
            stress_condition=no_stress_label,
            sponge=sponge,
            control_name=control_name,
            no_stress_label=no_stress_label,
        ),
        relevant_panel=_decision_card_panel_payload(
            row_idx=row_idx,
            sample_panel=relevant_sample,
            control_panel=relevant_control,
            sensor=sensor,
            stress_condition=stress,
            sponge=sponge,
            control_name=control_name,
            no_stress_label=no_stress_label,
        ),
        summary_strip=_decision_card_summary_strip_payload(
            support_row=support_row,
            panel_trace=relevant_sample,
        ),
    )


def _decision_card_panel_payload(
    *,
    row_idx: int,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    sensor: str,
    stress_condition: str,
    sponge: str,
    control_name: str,
    no_stress_label: str,
) -> _DecisionCardPanelPayload:
    panel_trace = pd.concat([sample_panel, control_panel], ignore_index=True)
    title = _decision_card_panel_title(stress_condition=stress_condition)
    return _DecisionCardPanelPayload(
        stress_condition=stress_condition,
        title=title,
        show_ylabel=stress_condition == no_stress_label,
        show_legend=stress_condition == no_stress_label,
        panel_trace=panel_trace,
        line_payloads=_decision_card_line_payloads(
            sample_panel=sample_panel,
            control_panel=control_panel,
            control_name=control_name,
        ),
    )


def _decision_card_panel_title(*, stress_condition: str) -> str:
    return str(stress_condition).strip()


def _decision_card_line_payloads(
    *,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    control_name: str,
) -> tuple[_DecisionCardTraceLinePayload, ...]:
    line_specs = (
        (
            "sample-minus",
            "On-target sponge -IPTG",
            sample_panel[sample_panel["IPTG"].astype(str) == "-IPTG"],
            "#1f77b4",
            "-",
        ),
        (
            "sample-plus",
            "On-target sponge +IPTG",
            sample_panel[sample_panel["IPTG"].astype(str) == "+IPTG"],
            "#1f77b4",
            "--",
        ),
        (
            "control-minus",
            f"matched {control_name} -IPTG",
            control_panel[control_panel["IPTG"].astype(str) == "-IPTG"],
            "#8c8c8c",
            "-",
        ),
        (
            "control-plus",
            f"matched {control_name} +IPTG",
            control_panel[control_panel["IPTG"].astype(str) == "+IPTG"],
            "#8c8c8c",
            "--",
        ),
    )
    keyed_frames: list[pd.DataFrame] = []
    keyed_styles: list[tuple[str, str, str, str]] = []
    for key, label, line_df, color, linestyle in line_specs:
        if line_df.empty:
            continue
        keyed_frames.append(line_df.assign(__line_key=key))
        keyed_styles.append((key, label, color, linestyle))
    if not keyed_frames:
        return ()
    summary_map = grouped_trace_summary_frames(
        pd.concat(keyed_frames, ignore_index=True),
        group_columns=("__line_key",),
    )
    return tuple(
        _DecisionCardTraceLinePayload(
            label=label,
            color=color,
            linestyle=linestyle,
            summary=summary_map.get((key,), empty_trace_summary_frame()),
        )
        for key, label, color, linestyle in keyed_styles
    )


def _decision_card_summary_strip_payload(
    *,
    support_row: Mapping[str, object],
    panel_trace: pd.DataFrame,
) -> _DecisionCardSummaryStripPayload:
    return _DecisionCardSummaryStripPayload(
        items=tuple(
            _decision_card_summary_item(
                spec=spec,
                support_row=support_row,
                panel_trace=panel_trace,
            )
            for spec in retron_presentation.decision_card_metric_specs()
        )
    )


def _decision_card_summary_item(
    *,
    spec: retron_presentation.DecisionCardMetricSpec,
    support_row: Mapping[str, object],
    panel_trace: pd.DataFrame,
) -> _DecisionCardSummaryItemPayload:
    minus_values = _support_tuple_values(support_row.get(f"{spec.metric}_minus_values"))
    plus_values = _support_tuple_values(support_row.get(f"{spec.metric}_plus_values"))
    contrast_values = _support_tuple_values(support_row.get(f"{spec.metric}_contrast_values"))
    minus_mean, minus_lower, minus_upper = _summary_strip_state_interval(minus_values)
    plus_mean, plus_lower, plus_upper = _summary_strip_state_interval(plus_values)
    contrast_mean = _support_float(support_row.get(f"{spec.metric}_mean"))
    contrast_lower = _support_float(support_row.get(f"{spec.metric}_lower"))
    contrast_upper = _support_float(support_row.get(f"{spec.metric}_upper"))
    finite = np.asarray(
        [*minus_values, *plus_values, *contrast_values, contrast_mean, contrast_lower, contrast_upper],
        dtype=float,
    )
    finite = finite[np.isfinite(finite)]
    limits = shared_numeric_limits(
        finite if finite.size else np.array([0.0], dtype=float),
        center=0.0,
        pad_fraction=0.12,
        min_span=0.10,
    )
    return _DecisionCardSummaryItemPayload(
        metric=spec.metric,
        title=retron_presentation.decision_card_metric_title(
            spec.metric,
            trace=panel_trace,
            summary_window_start_h=_support_float(support_row.get("summary_window_start_h")),
            summary_window_end_h=_support_float(support_row.get("summary_window_end_h")),
        ),
        label=spec.label,
        units=spec.units,
        axis_label=spec.axis_label,
        point_color=spec.color,
        minus_values=tuple(float(value) for value in minus_values if np.isfinite(value)),
        plus_values=tuple(float(value) for value in plus_values if np.isfinite(value)),
        contrast_values=tuple(float(value) for value in contrast_values if np.isfinite(value)),
        minus_mean=minus_mean,
        minus_lower=minus_lower,
        minus_upper=minus_upper,
        plus_mean=plus_mean,
        plus_lower=plus_lower,
        plus_upper=plus_upper,
        contrast_mean=contrast_mean,
        contrast_lower=contrast_lower,
        contrast_upper=contrast_upper,
        x_limits=limits,
    )


def _support_tuple_values(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray([], dtype=float)
    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple)):
        raw = np.asarray(value, dtype=float)
    else:
        raw = np.asarray([value], dtype=float)
    finite = raw[np.isfinite(raw)]
    return np.asarray(finite, dtype=float)


def _support_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if np.isfinite(numeric):
        return float(numeric)
    return None


def _summary_strip_metric_series(
    *,
    metric: str,
    sensor_trace: pd.DataFrame,
    sensor: str,
    sponge: str,
    stress: str,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    metric_id = str(metric)
    if metric_id == "P_pre":
        return _preload_state_series(sample_panel=sample_panel, control_panel=control_panel)
    if metric_id == "O_AUC":
        return _expected_direction_positive_state_area_series(
            sample_panel=sample_panel,
            control_panel=control_panel,
            sensor=sensor,
            sponge=sponge,
            stress=stress,
        )
    raise ValueError(f"retron_decomposition: unsupported summary-strip metric {metric_id!r}")


def _preload_state_series(
    *,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    minus = _state_preload_series(sample_panel=sample_panel, control_panel=control_panel, iptg="-IPTG")
    plus = _state_preload_series(sample_panel=sample_panel, control_panel=control_panel, iptg="+IPTG")
    return minus, plus


def _state_preload_series(
    *,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    iptg: str,
) -> pd.Series:
    sample = sample_panel[
        (sample_panel["IPTG"].astype(str) == str(iptg)) & sample_panel["in_pre_window"].fillna(False)
    ].copy()
    control = control_panel[
        (control_panel["IPTG"].astype(str) == str(iptg)) & control_panel["in_pre_window"].fillna(False)
    ].copy()
    if sample.empty or control.empty or "replicate_id" not in sample.columns:
        return pd.Series(dtype=float)
    sample_values = sample.groupby("replicate_id", dropna=False)["value"].mean().pipe(pd.to_numeric, errors="coerce")
    control_ref = pd.to_numeric(control["value"], errors="coerce").dropna().to_numpy(dtype=float)
    if control_ref.size == 0:
        return pd.Series(dtype=float)
    values = sample_values - float(control_ref.mean())
    return values.dropna().astype(float).sort_index()


def _absolute_effect_state_auc_series(
    *,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    minus = _state_adjusted_auc_series(sample_panel=sample_panel, control_panel=control_panel, iptg="-IPTG")
    plus = _state_adjusted_auc_series(sample_panel=sample_panel, control_panel=control_panel, iptg="+IPTG")
    return minus, plus


def _state_adjusted_auc_series(
    *,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    iptg: str,
) -> pd.Series:
    sample = sample_panel[
        (sample_panel["IPTG"].astype(str) == str(iptg)) & sample_panel["in_primary_post_stress"].fillna(False)
    ].copy()
    control = control_panel[
        (control_panel["IPTG"].astype(str) == str(iptg)) & control_panel["in_primary_post_stress"].fillna(False)
    ].copy()
    if sample.empty or control.empty or "replicate_id" not in sample.columns:
        return pd.Series(dtype=float)
    control_mean = (
        control.groupby("time_from_stress", dropna=False)["value"].mean().rename("control_value").reset_index()
    )
    values: dict[object, float] = {}
    for replicate_id, replicate in sample.groupby("replicate_id", dropna=False):
        ordered = replicate.sort_values("time_from_stress", kind="stable")
        aligned = ordered.merge(control_mean, on="time_from_stress", how="inner", validate="many_to_one")
        if aligned.empty:
            continue
        times = pd.to_numeric(aligned["time_from_stress"], errors="coerce").to_numpy(dtype=float)
        adjusted = pd.to_numeric(aligned["value"], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(
            aligned["control_value"], errors="coerce"
        ).to_numpy(dtype=float)
        values[replicate_id] = _auc(times, adjusted)
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float).dropna().sort_index()


def _matched_control_state_auc_series(
    *,
    sensor_trace: pd.DataFrame,
    sensor: str,
    sponge: str,
    stress: str,
    metric: str,
) -> tuple[pd.Series, pd.Series]:
    frame = sensor_trace[
        (sensor_trace["sensor"].astype(str) == str(sensor))
        & (sensor_trace["sponge"].astype(str) == str(sponge))
        & (sensor_trace["stress_condition"].astype(str) == str(stress))
        & (sensor_trace["metric"].astype(str) == str(metric))
        & sensor_trace["IPTG"].notna()
        & sensor_trace["in_primary_post_stress"].fillna(False)
    ].copy()
    if frame.empty:
        raise ValueError(
            "retron_decomposition: semantic trace input is missing replicate-level matched-control rows for "
            f"metric={metric!r}, sensor={sensor!r}, sponge={sponge!r}, stress={stress!r}"
        )
    return (
        _state_auc_series(frame=frame, iptg="-IPTG"),
        _state_auc_series(frame=frame, iptg="+IPTG"),
    )


def _state_auc_series(
    *,
    frame: pd.DataFrame,
    iptg: str,
) -> pd.Series:
    state = frame[frame["IPTG"].astype(str) == str(iptg)].copy()
    if state.empty or "replicate_id" not in state.columns:
        return pd.Series(dtype=float)
    values: dict[object, float] = {}
    for replicate_id, replicate in state.groupby("replicate_id", dropna=False):
        ordered = replicate.sort_values("time_from_stress", kind="stable")
        times = pd.to_numeric(ordered["time_from_stress"], errors="coerce").to_numpy(dtype=float)
        metric_values = pd.to_numeric(ordered["value"], errors="coerce").to_numpy(dtype=float)
        values[replicate_id] = _auc(times, metric_values)
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float).dropna().sort_index()


def _expected_direction_positive_state_area_series(
    *,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    sensor: str,
    sponge: str,
    stress: str,
) -> tuple[pd.Series, pd.Series]:
    return (
        _state_expected_direction_positive_area_series(
            sample_panel=sample_panel,
            control_panel=control_panel,
            sensor=sensor,
            sponge=sponge,
            stress=stress,
            iptg="-IPTG",
        ),
        _state_expected_direction_positive_area_series(
            sample_panel=sample_panel,
            control_panel=control_panel,
            sensor=sensor,
            sponge=sponge,
            stress=stress,
            iptg="+IPTG",
        ),
    )


def _state_expected_direction_positive_area_series(
    *,
    sample_panel: pd.DataFrame,
    control_panel: pd.DataFrame,
    sensor: str,
    sponge: str,
    stress: str,
    iptg: str,
) -> pd.Series:
    sample = sample_panel[
        (sample_panel["IPTG"].astype(str) == str(iptg)) & sample_panel["in_primary_post_stress"].fillna(False)
    ].copy()
    control = control_panel[
        (control_panel["IPTG"].astype(str) == str(iptg)) & control_panel["in_primary_post_stress"].fillna(False)
    ].copy()
    if "expected_decoy_sign" not in sample.columns:
        raise ValueError(
            "retron_decomposition: semantic trace input is missing 'expected_decoy_sign', which is required for "
            "expected-direction state-area summaries. Re-run the semantic metrics export before reopening the notebook."
        )
    if sample.empty or control.empty or "replicate_id" not in sample.columns:
        raise ValueError(
            "retron_decomposition: semantic trace input is missing replicate-level matched-control rows for "
            f"expected-direction state-area summaries, sensor={sensor!r}, sponge={sponge!r}, stress={stress!r}, "
            f"IPTG={iptg!r}"
        )
    control_mean = (
        control.groupby("time_from_stress", dropna=False)["value"].mean().rename("control_value").reset_index()
    )
    values: dict[object, float] = {}
    for replicate_id, replicate in sample.groupby("replicate_id", dropna=False):
        ordered = replicate.sort_values("time_from_stress", kind="stable")
        aligned = ordered.merge(control_mean, on="time_from_stress", how="inner", validate="many_to_one")
        if aligned.empty:
            continue
        times = pd.to_numeric(aligned["time_from_stress"], errors="coerce").to_numpy(dtype=float)
        metric_values = pd.to_numeric(aligned["value"], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(
            aligned["control_value"], errors="coerce"
        ).to_numpy(dtype=float)
        expected_sign = pd.to_numeric(ordered["expected_decoy_sign"], errors="coerce").dropna().to_numpy(dtype=float)
        if expected_sign.size == 0:
            raise ValueError(
                "retron_decomposition: semantic trace input has no finite 'expected_decoy_sign' values for "
                f"sensor={sensor!r}, sponge={sponge!r}, stress={stress!r}, IPTG={iptg!r}, "
                f"replicate_id={replicate_id!r}"
            )
        aligned = float(expected_sign[0]) * metric_values
        values[replicate_id] = _auc(times, np.maximum(aligned, 0.0))
    if not values:
        raise ValueError(
            "retron_decomposition: semantic trace input could not align matched-control rows for "
            f"expected-direction state-area summaries, sensor={sensor!r}, sponge={sponge!r}, stress={stress!r}, "
            f"IPTG={iptg!r}"
        )
    return pd.Series(values, dtype=float).dropna().sort_index()


def _paired_contrast_values(
    *,
    plus_series: pd.Series,
    minus_series: pd.Series,
) -> np.ndarray:
    if plus_series.empty or minus_series.empty:
        return np.asarray([], dtype=float)
    paired = plus_series.rename("plus").to_frame().join(minus_series.rename("minus"), how="inner")
    if paired.empty:
        return np.asarray([], dtype=float)
    contrast = pd.to_numeric(paired["plus"], errors="coerce") - pd.to_numeric(paired["minus"], errors="coerce")
    finite = contrast.dropna().to_numpy(dtype=float, copy=False)
    return np.asarray(finite, dtype=float)


def _summary_strip_contrast_interval(
    *,
    plus_values: np.ndarray,
    minus_values: np.ndarray,
    contrast_values: np.ndarray | None = None,
) -> tuple[float | None, float | None, float | None]:
    if contrast_values is not None:
        finite = np.asarray(contrast_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            return _summary_strip_state_interval(finite)
    if plus_values.size == 0 or minus_values.size == 0:
        return None, None, None
    mean, lower, upper = bootstrap_linear_interval(
        [plus_values, minus_values],
        coefficients=(1.0, -1.0),
        ci=95.0,
        ci_boot=200,
        rng=np.random.default_rng(0),
    )
    return float(mean), float(lower), float(upper)


def _summary_strip_state_interval(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    mean, lower, upper = bootstrap_mean_interval(
        values,
        ci=95.0,
        ci_boot=200,
        rng=np.random.default_rng(0),
    )
    return float(mean), float(lower), float(upper)


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _apply_shared_summary_x_limits(
    row_payloads: Sequence[_DecisionCardRowPayload],
) -> list[_DecisionCardRowPayload]:
    metric_limits: dict[str, tuple[float, float]] = {}
    for metric in (spec.metric for spec in retron_presentation.decision_card_metric_specs()):
        values: list[float] = []
        for row_payload in row_payloads:
            for item in row_payload.summary_strip.items:
                if item.metric != metric:
                    continue
                values.extend(item.minus_values)
                values.extend(item.plus_values)
                values.extend(item.contrast_values)
                for value in (
                    item.minus_mean,
                    item.minus_lower,
                    item.minus_upper,
                    item.plus_mean,
                    item.plus_lower,
                    item.plus_upper,
                    item.contrast_mean,
                    item.contrast_lower,
                    item.contrast_upper,
                ):
                    if value is not None and np.isfinite(value):
                        values.append(float(value))
        metric_limits[metric] = shared_numeric_limits(
            np.asarray(values if values else [0.0], dtype=float),
            center=0.0,
            pad_fraction=0.12,
            min_span=0.10,
        )
    updated_rows: list[_DecisionCardRowPayload] = []
    for row_payload in row_payloads:
        updated_items = tuple(
            replace(item, x_limits=metric_limits.get(item.metric, item.x_limits))
            for item in row_payload.summary_strip.items
        )
        updated_rows.append(replace(row_payload, summary_strip=replace(row_payload.summary_strip, items=updated_items)))
    return updated_rows


def _render_decision_card_row(
    *,
    figure: plt.Figure,
    row_spec,
    row: _DecisionCardRowPayload,
    display_bounds: tuple[float, float] | None,
    show_x_axis_labels: bool,
) -> None:
    summary_count = max(1, len(row.summary_strip.items))
    plot_column_count = 2 + summary_count
    group_grid = row_spec.subgridspec(
        1,
        plot_column_count + 1,
        width_ratios=(0.82, *([1.0] * plot_column_count)),
        wspace=0.18,
    )
    label_ax = figure.add_subplot(group_grid[0, 0])
    _render_decision_card_row_label(ax=label_ax, row=row)
    h2o_ax = figure.add_subplot(group_grid[0, 1])
    relevant_ax = figure.add_subplot(group_grid[0, 2], sharex=h2o_ax, sharey=h2o_ax)
    _plot_decision_card_trace_axis(
        ax=h2o_ax,
        panel=row.h2o_panel,
        panel_limits=row.panel_limits,
        display_bounds=display_bounds,
        show_x_axis_label=show_x_axis_labels,
    )
    _plot_decision_card_trace_axis(
        ax=relevant_ax,
        panel=row.relevant_panel,
        panel_limits=row.panel_limits,
        display_bounds=display_bounds,
        show_x_axis_label=show_x_axis_labels,
    )
    for col_idx, item in enumerate(row.summary_strip.items):
        _render_decision_card_metric_axis(
            ax=figure.add_subplot(group_grid[0, 3 + col_idx]),
            item=item,
            show_x_axis_label=show_x_axis_labels,
        )


def _render_decision_card_row_label(
    *,
    ax: plt.Axes,
    row: _DecisionCardRowPayload,
) -> None:
    ax.set_axis_off()
    ax.text(
        0.96,
        0.5,
        f"{row.sponge}\n{row.sensor}",
        ha="right",
        va="center",
        multialignment="right",
        fontsize=10.2,
        fontweight="bold",
        color="#222222",
        transform=ax.transAxes,
    )


def _plot_decision_card_trace_axis(
    *,
    ax: plt.Axes,
    panel: _DecisionCardPanelPayload,
    panel_limits: tuple[float, float],
    display_bounds: tuple[float, float] | None,
    show_x_axis_label: bool,
) -> None:
    legend_handles = _plot_decision_card_trace_lines(ax=ax, line_payloads=panel.line_payloads)
    _decorate_decision_card_trace_axis(
        ax=ax,
        panel=panel,
        panel_limits=panel_limits,
        display_bounds=display_bounds,
        legend_handles=legend_handles,
        show_x_axis_label=show_x_axis_label,
    )


def _plot_decision_card_trace_lines(
    *,
    ax: plt.Axes,
    line_payloads: Sequence[_DecisionCardTraceLinePayload],
) -> dict[str, object]:
    legend_handles: dict[str, object] = {}
    for line in line_payloads:
        _plot_trace_summary_line(ax=ax, line=line, legend_handles=legend_handles)
    return legend_handles


def _plot_trace_summary_line(
    *,
    ax: plt.Axes,
    line: _DecisionCardTraceLinePayload,
    legend_handles: dict[str, object],
) -> None:
    if line.summary.empty:
        return
    ax.fill_between(
        line.summary["time_from_stress"],
        line.summary["lower"],
        line.summary["upper"],
        alpha=0.16,
        color=line.color,
        linewidth=0.0,
        zorder=1,
    )
    (trace_line,) = ax.plot(
        line.summary["time_from_stress"].to_numpy(dtype=float),
        line.summary["mean"].to_numpy(dtype=float),
        color=line.color,
        linestyle=line.linestyle,
        linewidth=2.0,
        label=line.label,
        zorder=2,
    )
    legend_handles.setdefault(line.label, trace_line)


def _decorate_decision_card_trace_axis(
    *,
    ax: plt.Axes,
    panel: _DecisionCardPanelPayload,
    panel_limits: tuple[float, float],
    display_bounds: tuple[float, float] | None,
    legend_handles: Mapping[str, object],
    show_x_axis_label: bool,
) -> None:
    annotate_primary_window(ax, panel.panel_trace, stress_condition=panel.stress_condition)
    annotate_stress_addition(ax)
    ax.grid(
        axis="both",
        color=_DECISION_CARD_TRACE_AXIS_POLICY.grid_color,
        linewidth=_DECISION_CARD_TRACE_AXIS_POLICY.grid_linewidth,
        alpha=_DECISION_CARD_TRACE_AXIS_POLICY.grid_alpha,
    )
    ax.tick_params(axis="x", labelsize=_DECISION_CARD_TRACE_AXIS_POLICY.tick_size)
    ax.tick_params(axis="y", labelsize=_DECISION_CARD_TRACE_AXIS_POLICY.tick_size)
    ax.set_ylim(panel_limits)
    if display_bounds is not None:
        ax.set_xlim(display_bounds)
    if panel.title:
        ax.set_title(panel.title, pad=5, fontsize=9.6, fontweight="normal")
    if show_x_axis_label:
        ax.set_xlabel("Time from stress addition (h)", fontsize=8.9)
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(
        _DECISION_CARD_TRACE_AXIS_POLICY.ylabel if panel.show_ylabel else "",
        fontsize=_DECISION_CARD_TRACE_AXIS_POLICY.ylabel_fontsize,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_alpha(0.70)
    if panel.show_legend and legend_handles:
        ax.legend(
            frameon=False,
            title=None,
            loc=_DECISION_CARD_TRACE_AXIS_POLICY.legend_loc,
            fontsize=_DECISION_CARD_TRACE_AXIS_POLICY.legend_fontsize,
        )
    with suppress(Exception):
        ax.set_box_aspect(
            _DECISION_CARD_TRACE_AXIS_POLICY.box_aspect_with_ylabel
            if panel.show_ylabel
            else _DECISION_CARD_TRACE_AXIS_POLICY.box_aspect_without_ylabel
        )


def _render_decision_card_metric_axis(
    *,
    ax: plt.Axes,
    item: _DecisionCardSummaryItemPayload,
    show_x_axis_label: bool,
) -> None:
    ax.axvline(
        0.0,
        color=_DECISION_CARD_SUMMARY_AXIS_POLICY.zero_line_color,
        linewidth=_DECISION_CARD_SUMMARY_AXIS_POLICY.zero_line_linewidth,
        linestyle=_DECISION_CARD_SUMMARY_AXIS_POLICY.zero_line_linestyle,
        zorder=1,
    )
    ax.set_xlim(item.x_limits)
    if show_x_axis_label:
        ax.set_xlabel(item.axis_label, fontsize=_DECISION_CARD_SUMMARY_AXIS_POLICY.xlabel_fontsize)
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("", fontsize=_DECISION_CARD_SUMMARY_AXIS_POLICY.ylabel_fontsize)
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(["-IPTG", "+IPTG", "ΔIPTG"])
    ax.tick_params(axis="x", labelsize=_DECISION_CARD_SUMMARY_AXIS_POLICY.tick_size)
    ax.tick_params(axis="y", labelsize=_DECISION_CARD_SUMMARY_AXIS_POLICY.ytick_fontsize, pad=1.5)
    ax.grid(
        axis="x",
        color=_DECISION_CARD_SUMMARY_AXIS_POLICY.grid_color,
        linewidth=_DECISION_CARD_SUMMARY_AXIS_POLICY.grid_linewidth,
        alpha=_DECISION_CARD_SUMMARY_AXIS_POLICY.grid_alpha,
    )
    ax.set_title(
        item.title,
        loc="center",
        fontsize=_DECISION_CARD_SUMMARY_AXIS_POLICY.label_fontsize,
        pad=4,
        fontweight="normal",
    )
    _render_metric_state_points(ax=ax, values=item.minus_values, y_base=0.0, color="#8c8c8c")
    _render_metric_state_points(ax=ax, values=item.plus_values, y_base=1.0, color=item.point_color)
    _render_metric_state_points(ax=ax, values=item.contrast_values, y_base=2.0, color=item.point_color)
    _render_metric_state_estimate(
        ax=ax,
        value=item.minus_mean,
        lower=item.minus_lower,
        upper=item.minus_upper,
        y_base=0.0,
        color="#5f5f5f",
    )
    _render_metric_state_estimate(
        ax=ax,
        value=item.plus_mean,
        lower=item.plus_lower,
        upper=item.plus_upper,
        y_base=1.0,
        color=item.point_color,
    )
    if item.contrast_mean is None or not np.isfinite(item.contrast_mean):
        ax.text(0.5, 0.5, "NA", transform=ax.transAxes, ha="center", va="center", fontsize=8.0, color="#777777")
    else:
        _render_metric_state_estimate(
            ax=ax,
            value=item.contrast_mean,
            lower=item.contrast_lower,
            upper=item.contrast_upper,
            y_base=2.0,
            color=item.point_color,
        )
    ax.set_ylim(-0.5, 2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_alpha(0.70)
    with suppress(Exception):
        ax.set_box_aspect(_DECISION_CARD_SUMMARY_AXIS_POLICY.box_aspect)


def _decision_card_sensor_response_value(
    *,
    summary: pd.DataFrame,
    sensor: str,
    control_name: str,
) -> float:
    g_sensor = summary[
        (summary["sensor"].astype(str) == sensor)
        & (summary["sponge"].astype(str) == control_name)
        & (summary["metric"].astype(str) == "G_sensor")
    ]["value"]
    return _numeric_series_mean(g_sensor)


def _render_metric_state_points(
    *,
    ax: plt.Axes,
    values: Sequence[float],
    y_base: float,
    color: str,
) -> None:
    if not values:
        return
    jitter = _symmetric_jitter(len(values), span=0.16)
    ax.scatter(
        np.asarray(values, dtype=float),
        np.full(len(values), float(y_base), dtype=float) + jitter,
        s=22,
        color=color,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
        zorder=2.2,
    )


def _render_metric_state_estimate(
    *,
    ax: plt.Axes,
    value: float | None,
    lower: float | None,
    upper: float | None,
    y_base: float,
    color: str,
) -> None:
    if value is None or not np.isfinite(value):
        return
    estimate = float(value)
    low = estimate if lower is None or not np.isfinite(lower) else float(lower)
    high = estimate if upper is None or not np.isfinite(upper) else float(upper)
    xerr = np.asarray([[max(0.0, estimate - low)], [max(0.0, high - estimate)]], dtype=float)
    ax.errorbar(
        [estimate],
        [float(y_base)],
        xerr=xerr,
        fmt="o",
        markersize=6.8,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.2,
        ecolor=color,
        elinewidth=1.2,
        capsize=3.0,
        capthick=1.0,
        zorder=3.0,
    )


def _symmetric_jitter(count: int, *, span: float) -> np.ndarray:
    if count <= 1:
        return np.zeros(max(count, 0), dtype=float)
    return np.linspace(-float(span) / 2.0, float(span) / 2.0, count, dtype=float)


def _numeric_series_mean(values: pd.Series | object) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        numeric = numeric.dropna()
        return float(numeric.mean()) if not numeric.empty else float("nan")
    numeric_array = np.asarray(numeric, dtype=float)
    finite = numeric_array[np.isfinite(numeric_array)]
    return float(finite.mean()) if finite.size else float("nan")
