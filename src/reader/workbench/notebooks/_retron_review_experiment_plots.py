from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from reader.domains.plate_reader.plots.retron_sponge import (
    build_retron_decomposition_frame,
    plot_retron_sponge_summary,
    plot_retron_sponge_trace,
)
from reader.domains.plate_reader.plots.time_series import plot_time_series
from reader.workbench.notebooks import _retron_review_bundle as retron_review_bundle
from reader.workbench.notebooks import _retron_review_catalog as retron_review_catalog
from reader.workbench.notebooks import _retron_review_shared as retron_review_shared

RetronReviewSourceSurface = retron_review_bundle.RetronReviewSourceSurface

_normalize_optional_str = retron_review_shared.normalize_optional_str


@dataclass(frozen=True)
class ExperimentPlotPayload:
    plot_id: str
    with_cfg: dict[str, Any]
    figures: list[Any]
    supporting_table: pd.DataFrame
    supporting_table_title: str


@dataclass(frozen=True)
class SummaryPlotConfig:
    view: str
    title: str
    filename: str | None
    control_name: str
    no_stress_label: str
    relevant_only: bool
    metric: str | None
    state_order: list[str] | None
    burden_metric: str
    fig_kwargs: dict[str, Any]


@dataclass(frozen=True)
class _ExperimentPlotContext:
    plot_spec: Mapping[str, Any]
    plot_id: str
    plugin: str
    with_cfg: dict[str, Any]
    datasets: Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class _TimeSeriesPlotConfig:
    x: str
    xlabel: str | None
    y: list[str] | None
    ylabel_map: dict[str, str]
    hue_label_map: dict[str, str]
    hue: str
    channels: list[str] | None
    group_on: str | None
    pool_sets: Any
    pool_match: str
    fig_kwargs: dict[str, Any]
    add_sheet_line: bool
    sheet_line_kwargs: dict[str, Any]
    log_transform: Any
    time_window: list[float] | None
    ci: float
    ci_alpha: float
    ci_boot: int
    ci_seed: int
    legend_loc: str
    show_replicates: bool
    shared_legend: bool
    filename: str | None
    supporting_channels: list[str] | None


@dataclass(frozen=True)
class _TracePlotConfig:
    metrics: list[str]
    title: str
    filename: str | None
    control_name: str
    include_control: bool
    only_control: bool
    relevant_only: bool
    stress_order: list[str] | None
    panel_by: str
    metric_label_map: dict[str, str]
    fig_kwargs: dict[str, Any]


@dataclass(frozen=True)
class _SupportingTableSpec:
    keep_columns: tuple[str, ...]
    order_columns: tuple[str, ...]


_TIME_SERIES_SUPPORTING_TABLE_SPEC = _SupportingTableSpec(
    keep_columns=(
        "design_id_alias",
        "design_id",
        "treatment_alias",
        "treatment",
        "time",
        "sheet_index",
        "overflow",
        "channel",
        "value",
    ),
    order_columns=("design_id_alias", "design_id", "time", "channel"),
)

_TRACE_SUPPORTING_TABLE_SPEC = _SupportingTableSpec(
    keep_columns=(
        "sensor",
        "sponge",
        "stress_condition",
        "IPTG",
        "time_from_stress",
        "metric",
        "value",
        "matched_control_key",
        "configured_max_post_stress_hours",
        "summary_window_start_h",
        "summary_window_end_h",
        "summary_window_duration_h",
        "pre_stress_read_count",
        "post_stress_read_count",
        "matched_group_sample_count",
        "stress_addition_gap_h",
        "relevant_sensor_pair",
        "is_relevant_stress",
        "sponge_family_size",
    ),
    order_columns=("sensor", "sponge", "stress_condition", "IPTG", "time_from_stress", "metric"),
)

_SUMMARY_SUPPORTING_TABLE_SPEC = _SupportingTableSpec(
    keep_columns=(
        "sensor",
        "sponge",
        "stress_condition",
        "IPTG",
        "metric",
        "value",
        "matched_control_key",
        "summary_window_start_h",
        "summary_window_end_h",
        "summary_window_duration_h",
        "pre_stress_read_count",
        "post_stress_read_count",
        "matched_group_sample_count",
        "stress_addition_gap_h",
        "warning_flag",
        "scale_reference_abs_g_sensor",
        "scale_min_abs_g_sensor",
        "relevant_sensor_pair",
        "is_relevant_stress",
        "sponge_family_size",
    ),
    order_columns=("sensor", "sponge", "stress_condition", "IPTG", "metric"),
)

_DECISION_CARD_METRIC_ROWS = (
    ("P_pre", "Pre-stress contrast"),
    ("O_AUC", "Expected-direction state area"),
    ("D_growth_AUC", "Burden penalty"),
)

_DECOMPOSITION_TRACE_REQUIRED_COLUMNS = frozenset(
    {
        "matched_control_key",
        "summary_window_start_h",
        "summary_window_end_h",
        "summary_window_duration_h",
        "pre_stress_read_count",
        "post_stress_read_count",
        "matched_group_sample_count",
        "stress_addition_gap_h",
        "expected_decoy_sign",
    }
)


def render_experiment_plot_payload(
    plot_spec: Mapping[str, Any],
    *,
    datasets: Mapping[str, pd.DataFrame],
) -> ExperimentPlotPayload:
    context = _experiment_plot_context(plot_spec=plot_spec, datasets=datasets)
    return _experiment_payload_builder(context.plugin)(context)


def source_plot_spec(
    *,
    surface: RetronReviewSourceSurface,
    plot_id: str,
    label: str,
) -> Mapping[str, Any]:
    plot_id_value = str(plot_id).strip()
    plot_spec = next((spec for spec in surface.plot_specs if str(spec.get("id", "")) == plot_id_value), None)
    if plot_spec is None:
        raise ValueError(f"retron_review: unknown scoped plot id {plot_id_value!r} for source {label!r}")
    return plot_spec


def load_source_plot_datasets(
    *,
    surface: RetronReviewSourceSurface,
    plot_spec: Mapping[str, Any],
    load_frame: Callable[[str], pd.DataFrame],
    semantic_datasets: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    record_paths = dict(surface.record_paths)
    semantic_datasets = semantic_datasets or {}
    datasets: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for record_id in _source_plot_record_ids(plot_spec, record_paths=record_paths):
        if not record_id or record_id in datasets:
            continue
        if record_id in semantic_datasets:
            datasets[record_id] = semantic_datasets[record_id]
            continue
        record_path = record_paths.get(record_id)
        if record_path is None:
            errors.append(f"Missing dataframe record `{record_id}` for the selected source plot.")
            continue
        try:
            datasets[record_id] = load_frame(record_path)
        except Exception as exc:
            errors.append(f"Failed to load `{record_id}`: {exc}")
    if errors:
        raise ValueError(" ".join(errors))
    return datasets


def source_plot_record_ids(
    plot_spec: Mapping[str, Any],
    *,
    record_paths: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    return _source_plot_record_ids(plot_spec, record_paths=record_paths)


def summary_plot_config(
    *,
    plot_spec: Mapping[str, Any],
    with_cfg: Mapping[str, Any],
) -> SummaryPlotConfig:
    return SummaryPlotConfig(
        view=str(with_cfg.get("view") or "heatmap"),
        title=str(with_cfg.get("title") or "Retron sponge summary"),
        filename=_normalize_optional_str(with_cfg.get("filename")),
        control_name=str(with_cfg.get("control_name") or "tetO"),
        no_stress_label=str(with_cfg.get("no_stress_label") or "H2O"),
        relevant_only=bool(with_cfg.get("relevant_only", True)),
        metric=_normalize_optional_str(with_cfg.get("metric")),
        state_order=_normalize_optional_str_list(with_cfg.get("state_order")),
        burden_metric=str(with_cfg.get("burden_metric") or retron_review_catalog.DEFAULT_RETRON_BURDEN_METRIC),
        fig_kwargs=dict(_validated_plot_mapping(plot_spec=plot_spec, field="with.fig", value=with_cfg.get("fig"))),
    )


def summary_supporting_table(
    summary_df: pd.DataFrame,
    *,
    view: str,
    metric: str,
    burden_metric: str,
) -> pd.DataFrame:
    frame = summary_df.copy()
    metric_names = retron_review_catalog.summary_supporting_metrics(
        view=view,
        metric=metric,
        burden_metric=burden_metric,
    )
    if metric_names:
        frame = frame[frame["metric"].astype(str).isin(metric_names)].copy()
    return _project_supporting_table(frame, spec=_SUMMARY_SUPPORTING_TABLE_SPEC)


def _validated_plot_mapping(
    *,
    plot_spec: Mapping[str, Any],
    field: str,
    value: Any,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        plot_id = str(plot_spec.get("id") or "").strip() or "<unknown>"
        raise ValueError(f"retron_review: plot {plot_id!r} field {field!r} must be a mapping")
    return dict(value)


def _experiment_plot_context(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> _ExperimentPlotContext:
    plot_id = str(plot_spec.get("id") or "").strip()
    if not plot_id:
        raise ValueError("retron_review: plot spec is missing an id")
    return _ExperimentPlotContext(
        plot_spec=plot_spec,
        plot_id=plot_id,
        plugin=str(plot_spec.get("plugin") or "").strip(),
        with_cfg=_validated_plot_mapping(plot_spec=plot_spec, field="with", value=plot_spec.get("with")),
        datasets=datasets,
    )


def _experiment_payload_builder(plugin: str) -> Callable[[_ExperimentPlotContext], ExperimentPlotPayload]:
    builders: dict[str, Callable[[_ExperimentPlotContext], ExperimentPlotPayload]] = {
        "plot/time_series": _render_time_series_notebook_payload,
        "plot/retron_trace": _render_trace_notebook_payload,
        "plot/retron_summary": _render_summary_notebook_payload,
    }
    try:
        return builders[str(plugin)]
    except KeyError as exc:
        raise ValueError(f"retron_review: unsupported notebook plot plugin {plugin!r}") from exc


def _render_time_series_notebook_payload(context: _ExperimentPlotContext) -> ExperimentPlotPayload:
    if context.plot_id in retron_review_catalog.RETRON_QC_PLOT_IDS:
        qc_df = _retron_qc_dataframe(plot_spec=context.plot_spec, datasets=context.datasets)
        config = _qc_time_series_plot_config(plot_spec=context.plot_spec, with_cfg=context.with_cfg, df=qc_df)
        return ExperimentPlotPayload(
            plot_id=context.plot_id,
            with_cfg=context.with_cfg,
            figures=_render_time_series_figures(
                df=qc_df,
                blanks=_plot_blanks_dataset(plot_spec=context.plot_spec, datasets=context.datasets, like=qc_df),
                config=config,
            ),
            supporting_table=_time_series_supporting_table(qc_df, channels=config.supporting_channels),
            supporting_table_title=(
                "Underlying overflow-handled raw channel rows plus derived support-ratio rows for the selected QC view"
            ),
        )
    config = _time_series_plot_config(plot_spec=context.plot_spec, with_cfg=context.with_cfg)
    df = _require_plot_dataset(plot_spec=context.plot_spec, datasets=context.datasets, label="df")
    return ExperimentPlotPayload(
        plot_id=context.plot_id,
        with_cfg=context.with_cfg,
        figures=_render_time_series_figures(
            df=df,
            blanks=_plot_blanks_dataset(plot_spec=context.plot_spec, datasets=context.datasets, like=df),
            config=config,
        ),
        supporting_table=_time_series_supporting_table(df, channels=config.supporting_channels),
        supporting_table_title="Underlying tidy rows for the selected raw or support channels",
    )


def _render_trace_notebook_payload(context: _ExperimentPlotContext) -> ExperimentPlotPayload:
    config = _trace_plot_config(plot_spec=context.plot_spec, with_cfg=context.with_cfg)
    trace_df = _require_plot_dataset(plot_spec=context.plot_spec, datasets=context.datasets, label="trace")
    return ExperimentPlotPayload(
        plot_id=context.plot_id,
        with_cfg=context.with_cfg,
        figures=_render_trace_figures(trace=trace_df, config=config),
        supporting_table=_trace_supporting_table(trace_df, metrics=config.metrics),
        supporting_table_title="Underlying assay trace rows for the selected kinetic transform",
    )


def _render_summary_notebook_payload(context: _ExperimentPlotContext) -> ExperimentPlotPayload:
    config = summary_plot_config(plot_spec=context.plot_spec, with_cfg=context.with_cfg)
    summary = _require_plot_dataset(plot_spec=context.plot_spec, datasets=context.datasets, label="summary")
    trace = _optional_plot_dataset(plot_spec=context.plot_spec, datasets=context.datasets, label="trace")
    _validate_summary_plot_inputs(plot_id=context.plot_id, summary=summary, trace=trace, config=config)
    figures = _render_summary_figures(summary=summary, trace=trace, config=config)
    if config.view == "decomposition":
        support_table = build_retron_decomposition_frame(
            _require_plot_dataset(plot_spec=context.plot_spec, datasets=context.datasets, label="trace"),
            control_name=config.control_name,
            relevant_only=config.relevant_only,
            summary=summary,
            no_stress_label=config.no_stress_label,
        )
        return ExperimentPlotPayload(
            plot_id=context.plot_id,
            with_cfg=context.with_cfg,
            figures=figures,
            supporting_table=_decision_card_supporting_table(support_table),
            supporting_table_title="Matched-tetO summary rows with QC checks, window metadata, and interval estimates",
        )
    return ExperimentPlotPayload(
        plot_id=context.plot_id,
        with_cfg=context.with_cfg,
        figures=figures,
        supporting_table=summary_supporting_table(
            summary,
            view=config.view,
            metric=config.metric or "",
            burden_metric=config.burden_metric,
        ),
        supporting_table_title="Underlying assay summary rows for the selected ranking view",
    )


def _validate_summary_plot_inputs(
    *,
    plot_id: str,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    config: SummaryPlotConfig,
) -> None:
    if config.view != "decomposition":
        return
    _validate_decomposition_trace_contract(plot_id=plot_id, trace=trace)
    _validate_decomposition_summary_contract(plot_id=plot_id, summary=summary, burden_metric=config.burden_metric)


def _validate_decomposition_trace_contract(
    *,
    plot_id: str,
    trace: pd.DataFrame | None,
) -> None:
    if trace is None:
        raise ValueError(
            "retron_review: the sponge-versus-matched-tetO summary requires both semantic summary and semantic trace records. "
            "Re-run `uv run reader run <experiment-config>` or refresh the dataframe records before reopening "
            "the notebook."
        )
    missing = sorted(_DECOMPOSITION_TRACE_REQUIRED_COLUMNS - set(trace.columns))
    if not missing:
        return
    raise ValueError(
        f"retron_review: plot {plot_id!r} requires refreshed semantic trace records for the sponge-versus-matched-tetO summary. "
        f"Missing columns: {missing}. The current dataframe record is stale relative to the retron notebook "
        "contract. Re-run `uv run reader run <experiment-config>` or regenerate the dataframe records before "
        "reopening the notebook."
    )


def _validate_decomposition_summary_contract(
    *,
    plot_id: str,
    summary: pd.DataFrame,
    burden_metric: str,
) -> None:
    if "metric" not in summary.columns:
        raise ValueError(
            f"retron_review: plot {plot_id!r} requires semantic summary rows with a 'metric' column for the "
            "sponge-versus-matched-tetO summary. Re-run `uv run reader run <experiment-config>` or regenerate "
            "the dataframe records before reopening the notebook."
        )
    available = {str(value) for value in summary["metric"].dropna().astype(str)}
    required = ("P_pre", "O_AUC", str(burden_metric), "G_sensor")
    missing = [metric for metric in required if metric not in available]
    if not missing:
        return
    raise ValueError(
        f"retron_review: plot {plot_id!r} requires matched-tetO summary metrics {missing}, but they are absent "
        "from the loaded semantic summary record. The current dataframe record is stale relative to the retron "
        "notebook contract. Re-run `uv run reader run <experiment-config>` or regenerate the dataframe records "
        "before reopening the notebook."
    )


def _plot_blanks_dataset(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
    like: pd.DataFrame,
) -> pd.DataFrame:
    blanks = _optional_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="blanks")
    return blanks if blanks is not None else like.iloc[0:0].copy()


def _time_series_plot_config(
    *,
    plot_spec: Mapping[str, Any],
    with_cfg: Mapping[str, Any],
) -> _TimeSeriesPlotConfig:
    partition = _validated_plot_mapping(
        plot_spec=plot_spec,
        field="with.partition",
        value=with_cfg.get("partition"),
    )
    y_channels = _normalize_optional_str_list(with_cfg.get("y"))
    channels = _normalize_optional_str_list(with_cfg.get("channels"))
    return _TimeSeriesPlotConfig(
        x=str(with_cfg.get("x") or "time"),
        xlabel=_normalize_optional_str(with_cfg.get("xlabel")),
        y=y_channels,
        ylabel_map=dict(
            _validated_plot_mapping(plot_spec=plot_spec, field="with.ylabel_map", value=with_cfg.get("ylabel_map"))
        ),
        hue_label_map=dict(
            _validated_plot_mapping(
                plot_spec=plot_spec,
                field="with.hue_label_map",
                value=with_cfg.get("hue_label_map"),
            )
        ),
        hue=str(with_cfg.get("hue") or "treatment"),
        channels=channels,
        group_on=_normalize_optional_str(partition.get("by")),
        pool_sets=partition.get("collection_items"),
        pool_match=str(partition.get("match") or "exact"),
        fig_kwargs=dict(_validated_plot_mapping(plot_spec=plot_spec, field="with.fig", value=with_cfg.get("fig"))),
        add_sheet_line=bool(with_cfg.get("add_sheet_line", False)),
        sheet_line_kwargs=dict(
            _validated_plot_mapping(
                plot_spec=plot_spec,
                field="with.sheet_line_kwargs",
                value=with_cfg.get("sheet_line_kwargs"),
            )
        ),
        log_transform=with_cfg.get("log_transform", False),
        time_window=_normalize_optional_float_list(with_cfg.get("time_window")),
        ci=float(with_cfg.get("ci", 95.0)),
        ci_alpha=float(with_cfg.get("ci_alpha", 0.15)),
        ci_boot=int(with_cfg.get("ci_boot", 100)),
        ci_seed=int(with_cfg.get("ci_seed", 0)),
        legend_loc=str(with_cfg.get("legend_loc") or "upper left"),
        show_replicates=bool(with_cfg.get("show_replicates", False)),
        shared_legend=bool(with_cfg.get("shared_legend", False)),
        filename=_normalize_optional_str(with_cfg.get("filename")),
        supporting_channels=y_channels or channels,
    )


def _qc_time_series_plot_config(
    *,
    plot_spec: Mapping[str, Any],
    with_cfg: Mapping[str, Any],
    df: pd.DataFrame,
) -> _TimeSeriesPlotConfig:
    preferred_channels = ["OD600", "YFP", "CFP", "YFP/CFP"]
    available = set(df["channel"].astype(str).tolist()) if "channel" in df.columns else set()
    channels = [channel for channel in preferred_channels if channel in available]
    config = _time_series_plot_config(plot_spec=plot_spec, with_cfg=with_cfg)
    ylabel_map = {
        "OD600": "OD600",
        "YFP": "YFP",
        "CFP": "CFP",
        "YFP/OD600": "YFP/OD600",
        "CFP/OD600": "CFP/OD600",
        "YFP/CFP": "YFP/CFP",
    }
    return replace(
        config,
        xlabel=str(with_cfg.get("xlabel") or "Time from stress addition (h)"),
        y=channels,
        ylabel_map={key: value for key, value in ylabel_map.items() if key in channels},
        channels=None,
        add_sheet_line=True,
        sheet_line_kwargs={"color": "#9E9E9E", "linestyle": "--", "linewidth": 0.9, "alpha": 0.95},
        shared_legend=True,
        supporting_channels=channels,
    )


def _render_time_series_figures(
    *,
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    config: _TimeSeriesPlotConfig,
) -> list[Any]:
    return plot_time_series(
        df=df,
        blanks=blanks,
        output_dir=None,
        x=config.x,
        xlabel=config.xlabel,
        y=config.y,
        ylabel_map=config.ylabel_map,
        hue_label_map=config.hue_label_map,
        hue=config.hue,
        channels=config.channels,
        subplots=None,
        group_on=config.group_on,
        pool_sets=config.pool_sets,
        pool_match=config.pool_match,
        fig_kwargs=config.fig_kwargs,
        add_sheet_line=config.add_sheet_line,
        sheet_line_kwargs=config.sheet_line_kwargs,
        log_transform=config.log_transform,
        time_window=config.time_window,
        palette_book=None,
        ci=config.ci,
        ci_alpha=config.ci_alpha,
        ci_boot=config.ci_boot,
        ci_seed=config.ci_seed,
        legend_loc=config.legend_loc,
        show_replicates=config.show_replicates,
        shared_legend=config.shared_legend,
        filename=config.filename,
    )


def _retron_qc_dataframe(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    ratio_df = _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="df")
    raw_df = datasets.get("overflow/df")
    if raw_df is None or raw_df.empty or "channel" not in raw_df.columns:
        return ratio_df
    raw_channels = [channel for channel in raw_df["channel"].astype(str).unique().tolist() if "/" not in channel]
    if not raw_channels:
        return ratio_df
    support_channels = (
        [channel for channel in ratio_df["channel"].astype(str).unique().tolist() if "/" in channel]
        if "channel" in ratio_df.columns
        else []
    )
    frames = [raw_df[raw_df["channel"].astype(str).isin(raw_channels)].copy()]
    if support_channels:
        frames.append(ratio_df[ratio_df["channel"].astype(str).isin(support_channels)].copy())
    all_columns = list(dict.fromkeys([*raw_df.columns.tolist(), *ratio_df.columns.tolist()]))
    aligned = [frame.reindex(columns=all_columns) for frame in frames if not frame.empty]
    if not aligned:
        return ratio_df
    return pd.concat(aligned, ignore_index=True)


def _trace_plot_config(
    *,
    plot_spec: Mapping[str, Any],
    with_cfg: Mapping[str, Any],
) -> _TracePlotConfig:
    return _TracePlotConfig(
        metrics=_normalize_optional_str_list(with_cfg.get("metrics")) or [],
        title=str(with_cfg.get("title") or "Retron sponge trace"),
        filename=_normalize_optional_str(with_cfg.get("filename")),
        control_name=str(with_cfg.get("control_name") or "tetO"),
        include_control=bool(with_cfg.get("include_control", False)),
        only_control=bool(with_cfg.get("only_control", False)),
        relevant_only=bool(with_cfg.get("relevant_only", False)),
        stress_order=_normalize_optional_str_list(with_cfg.get("stress_order")),
        panel_by=str(with_cfg.get("panel_by") or "stress"),
        metric_label_map=dict(
            _validated_plot_mapping(
                plot_spec=plot_spec,
                field="with.metric_label_map",
                value=with_cfg.get("metric_label_map"),
            )
        ),
        fig_kwargs=dict(_validated_plot_mapping(plot_spec=plot_spec, field="with.fig", value=with_cfg.get("fig"))),
    )


def _render_trace_figures(*, trace: pd.DataFrame, config: _TracePlotConfig) -> list[Any]:
    return plot_retron_sponge_trace(
        trace=trace,
        output_dir=None,
        metrics=config.metrics,
        title=config.title,
        filename=config.filename,
        palette_book=None,
        control_name=config.control_name,
        include_control=config.include_control,
        only_control=config.only_control,
        relevant_only=config.relevant_only,
        stress_order=config.stress_order,
        panel_by=config.panel_by,
        metric_label_map=config.metric_label_map,
        fig_kwargs=config.fig_kwargs,
    )


def _render_summary_figures(
    *,
    summary: pd.DataFrame,
    trace: pd.DataFrame | None,
    config: SummaryPlotConfig,
) -> list[Any]:
    return plot_retron_sponge_summary(
        summary=summary,
        trace=trace,
        output_dir=None,
        view=config.view,
        title=config.title,
        filename=config.filename,
        palette_book=None,
        control_name=config.control_name,
        no_stress_label=config.no_stress_label,
        relevant_only=config.relevant_only,
        metric=config.metric,
        state_order=config.state_order,
        burden_metric=config.burden_metric,
        fig_kwargs=config.fig_kwargs,
    )


def _require_plot_dataset(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    record_id = _read_record_id(plot_spec=plot_spec, label=label)
    try:
        return datasets[record_id]
    except KeyError as exc:
        raise ValueError(
            f"retron_review: plot {plot_spec.get('id')!r} requires record {record_id!r} for input {label!r}"
        ) from exc


def _optional_plot_dataset(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
    label: str,
) -> pd.DataFrame | None:
    record_id = _read_record_id(plot_spec=plot_spec, label=label, optional=True)
    if record_id is None:
        return None
    return datasets.get(record_id)


def _source_plot_record_ids(
    plot_spec: Mapping[str, Any],
    *,
    record_paths: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    record_ids = [record_id for record_id in _declared_plot_record_ids(plot_spec) if record_id]
    available_paths = record_paths or {}
    if (
        str(plot_spec.get("id") or "").strip() in retron_review_catalog.RETRON_QC_PLOT_IDS
        and "overflow/df" in available_paths
    ):
        record_ids.append("overflow/df")
    return tuple(dict.fromkeys(record_ids))


def _declared_plot_record_ids(plot_spec: Mapping[str, Any]) -> tuple[str, ...]:
    reads = _validated_plot_mapping(plot_spec=plot_spec, field="reads", value=plot_spec.get("reads"))
    record_ids: list[str] = []
    for label, read_ref in reads.items():
        binding = _validated_plot_mapping(
            plot_spec=plot_spec,
            field=f"reads.{label}",
            value=read_ref,
        )
        record_ids.append(str(binding.get("record", "")).strip())
    return tuple(record_ids)


def _read_record_id(
    *,
    plot_spec: Mapping[str, Any],
    label: str,
    optional: bool = False,
) -> str | None:
    reads = _validated_plot_mapping(plot_spec=plot_spec, field="reads", value=plot_spec.get("reads"))
    ref = _validated_plot_mapping(plot_spec=plot_spec, field=f"reads.{label}", value=reads.get(label))
    record_id = _normalize_optional_str(ref.get("record"))
    if record_id is None:
        if optional:
            return None
        raise ValueError(f"retron_review: plot {plot_spec.get('id')!r} is missing a record binding for {label!r}")
    return record_id


def _time_series_supporting_table(df: pd.DataFrame, *, channels: list[str] | None) -> pd.DataFrame:
    frame = df.copy()
    if channels and "channel" in frame.columns:
        frame = frame[frame["channel"].astype(str).isin(channels)].copy()
    return _project_supporting_table(frame, spec=_TIME_SERIES_SUPPORTING_TABLE_SPEC)


def _trace_supporting_table(trace_df: pd.DataFrame, *, metrics: list[str]) -> pd.DataFrame:
    frame = trace_df.copy()
    if metrics:
        frame = frame[frame["metric"].astype(str).isin(metrics)].copy()
    return _project_supporting_table(frame, spec=_TRACE_SUPPORTING_TABLE_SPEC)


def _decision_card_supporting_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    base_columns = [
        column
        for column in (
            "sensor",
            "sponge",
            "primary_stress",
            "stress_condition",
            "panel_role",
            "matched_control_key",
            "matched_group_sample_count",
            "pre_stress_read_count",
            "post_stress_read_count",
            "summary_window_start_h",
            "summary_window_end_h",
            "summary_window_duration_h",
            "stress_addition_gap_h",
            "G_sensor",
            "warning_flag",
        )
        if column in frame.columns
    ]
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        base = {column: row.get(column) for column in base_columns}
        for metric_id, label in _DECISION_CARD_METRIC_ROWS:
            estimate = row.get(f"{metric_id}_mean")
            lower = row.get(f"{metric_id}_lower")
            upper = row.get(f"{metric_id}_upper")
            if pd.isna(estimate) and pd.isna(lower) and pd.isna(upper):
                continue
            record = dict(base)
            record.update(
                {
                    "summary_metric": "O_state_AUC" if metric_id == "O_AUC" else metric_id,
                    "summary_label": label,
                    "estimate": estimate,
                    "lower": lower,
                    "upper": upper,
                    "units": row.get(f"{metric_id}_units"),
                }
            )
            minus_n = row.get(f"{metric_id}_minus_n")
            plus_n = row.get(f"{metric_id}_plus_n")
            minus_mean = row.get(f"{metric_id}_minus_mean")
            plus_mean = row.get(f"{metric_id}_plus_mean")
            if not pd.isna(minus_n):
                record["minus_state_n"] = minus_n
            if not pd.isna(plus_n):
                record["plus_state_n"] = plus_n
            if not pd.isna(minus_mean):
                record["minus_state_mean"] = minus_mean
            if not pd.isna(plus_mean):
                record["plus_state_mean"] = plus_mean
            rows.append(record)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = [
        column
        for column in ("sensor", "sponge", "primary_stress", "panel_role", "stress_condition", "summary_metric")
        if column in out.columns
    ]
    if order:
        out = out.sort_values(order, kind="stable")
    return out.reset_index(drop=True)


def _project_supporting_table(frame: pd.DataFrame, *, spec: _SupportingTableSpec) -> pd.DataFrame:
    keep = [column for column in spec.keep_columns if column in frame.columns]
    out = frame[keep]
    order = [column for column in spec.order_columns if column in keep]
    if order:
        out = out.sort_values(order, kind="stable")
    return out


def _normalize_optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]
