from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from reader.domains.plate_reader.analysis import retron_review_aggregate
from reader.workbench.notebooks import _retron_review_aggregate_plots as retron_review_aggregate_plots
from reader.workbench.notebooks import _retron_review_bundle as retron_review_bundle
from reader.workbench.notebooks import _retron_review_catalog as retron_review_catalog
from reader.workbench.notebooks import _retron_review_copy as retron_review_copy
from reader.workbench.notebooks import _retron_review_experiment_plots as retron_review_experiment_plots
from reader.workbench.notebooks import _retron_review_notebook_ui as retron_review_notebook_ui
from reader.workbench.notebooks import context as notebook_context

load_notebook_workbench_context = notebook_context.load_notebook_workbench_context
RetronReviewSource = retron_review_bundle.RetronReviewSource
RetronReviewBundle = retron_review_bundle.RetronReviewBundle
RetronReviewSourceSurface = retron_review_bundle.RetronReviewSourceSurface

aggregate_on_target_scores = retron_review_aggregate.aggregate_on_target_scores
available_aggregate_score_metrics = retron_review_aggregate.available_aggregate_score_metrics
available_multifunctional_sponges = retron_review_aggregate.available_multifunctional_sponges
build_aggregate_pareto_frame = retron_review_aggregate.build_aggregate_pareto_frame
build_architecture_frame = retron_review_aggregate.build_architecture_frame
build_expected_vs_observed_frame = retron_review_aggregate.build_expected_vs_observed_frame
build_fingerprint_frame = retron_review_aggregate.build_fingerprint_frame
build_specificity_matrix = retron_review_aggregate.build_specificity_matrix

load_cached_parquet_frame = retron_review_bundle.load_cached_parquet_frame
load_retron_source_semantic_datasets = retron_review_bundle.load_retron_source_semantic_datasets
load_retron_review_bundle = retron_review_bundle.load_retron_review_bundle
load_retron_semantic_maps_from_config = retron_review_bundle.load_retron_semantic_maps_from_config
load_retron_source_record_frame = retron_review_bundle.load_retron_source_record_frame
load_retron_source_surface = retron_review_bundle.load_retron_source_surface
retron_plot_rendered_files = retron_review_bundle.retron_plot_rendered_files
retron_visible_plot_specs = retron_review_bundle.retron_visible_plot_specs
contextualize_retron_plot_copy = retron_review_copy.contextualize_retron_plot_copy
dataframe_to_csv_bytes = retron_review_notebook_ui.dataframe_to_csv_bytes
download_safe_stem = retron_review_notebook_ui.download_safe_stem
figure_to_download_bytes = retron_review_notebook_ui.figure_to_download_bytes
filter_supporting_table_for_figure = retron_review_notebook_ui.filter_supporting_table_for_figure
retron_figure_label = retron_review_notebook_ui.retron_figure_label
retron_notebook_table_preview = retron_review_notebook_ui.retron_notebook_table_preview

_prepare_notebook_plot_figure = retron_review_notebook_ui.prepare_notebook_plot_figure
_style_notebook_figure = retron_review_notebook_ui.style_notebook_figure


@dataclass(frozen=True)
class RetronNotebookPlotResult:
    plot_id: str
    title: str
    stage: str
    question: str
    math: str
    meaning: str
    source_record: str
    figures: tuple[Any, ...]
    supporting_table: pd.DataFrame
    supporting_table_title: str


@dataclass(frozen=True)
class RetronAggregatePlotResult:
    plot_id: str
    title: str
    question: str
    math: str
    meaning: str
    figure: Any | None
    supporting_table: pd.DataFrame
    supporting_table_title: str


def retron_transform_ladder_rows() -> list[dict[str, str]]:
    return retron_review_catalog.retron_transform_ladder_rows()


def retron_aggregate_figure_rows() -> list[dict[str, str]]:
    return retron_review_catalog.retron_aggregate_figure_rows()


def retron_aggregate_plot_rows(plot_ids: list[str] | None = None) -> list[dict[str, str]]:
    selected = sorted(
        plot_ids or retron_review_aggregate_plots.aggregate_plot_specs(),
        key=retron_review_catalog.aggregate_plot_display_order,
    )
    rows: list[dict[str, str]] = []
    for plot_id in selected:
        guide = retron_review_aggregate_plots.aggregate_plot_spec(str(plot_id)).guide
        rows.append(
            {
                "Plot id": str(plot_id),
                "Figure": guide.title,
                "Math / transform": guide.math,
                "How to read": guide.meaning,
            }
        )
    return rows


def retron_experiment_plot_rows(plot_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in plot_specs:
        plot_id = str(spec.get("id", "")).strip()
        if not plot_id:
            continue
        guide = _experiment_plot_guide(plot_id)
        with_cfg = spec.get("with") if isinstance(spec.get("with"), Mapping) else {}
        fallback_title = str(with_cfg.get("title") or plot_id)
        title = str(guide.selector_title or guide.title or fallback_title)
        rows.append(
            {
                "Selector label": title,
                "Stage": guide.stage,
                "Plot": title,
                "Plot id": plot_id,
                "Display order": retron_review_catalog.experiment_plot_display_order(plot_id),
                "Math / transform": guide.math,
                "How to read": guide.meaning,
            }
        )
    rows.sort(key=lambda row: (int(row["Display order"]), str(row["Plot"])))
    return rows


def build_label_value_options(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_key: str,
    value_key: str,
    disambiguator_key: str | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(label_key, "")).strip()
        counts[label] = counts.get(label, 0) + 1
    options: dict[str, Any] = {}
    for row in rows:
        label = str(row.get(label_key, "")).strip()
        if counts.get(label, 0) > 1:
            disambiguator = str(row.get(disambiguator_key or value_key, "")).strip()
            label = f"{label} [{disambiguator}]"
        if label in options:
            raise ValueError(f"retron_review: duplicate selector label {label!r}")
        options[label] = row.get(value_key)
    return options


def retron_table_kwargs(
    *,
    page_size: int | None = None,
    pagination: bool | None = None,
    wrapped_columns: Sequence[str] | None = None,
    max_height: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "selection": None,
        "show_column_summaries": False,
        "show_data_types": False,
        "show_download": False,
    }
    if page_size is not None:
        kwargs["page_size"] = page_size
    if pagination is not None:
        kwargs["pagination"] = pagination
    if wrapped_columns:
        kwargs["wrapped_columns"] = list(wrapped_columns)
    if max_height is not None:
        kwargs["max_height"] = max_height
    return kwargs


def retron_figure_coverage_rows() -> list[dict[str, str]]:
    return retron_review_catalog.retron_figure_coverage_rows()


def retron_plot_guide_rows(plot_ids: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for plot_id in plot_ids:
        guide = _experiment_plot_guide(str(plot_id))
        rows.append(
            {
                "Stage": guide.stage,
                "Plot": guide.title,
                "Plot id": str(plot_id),
                "Math / transform": guide.math,
                "Source record": guide.record,
                "How to read": guide.meaning,
            }
        )
    rows.sort(key=lambda row: (_plot_stage_rank(row["Stage"]), row["Plot"]))
    return rows


def retron_assay_context_rows() -> list[dict[str, str]]:
    return retron_review_catalog.retron_assay_context_rows()


def retron_sensor_context_rows(
    relevant_stress_map: Mapping[str, str],
    sensor_target_map: Mapping[str, tuple[str, ...]],
) -> list[dict[str, str]]:
    sensors = sorted({str(key) for key in relevant_stress_map} | {str(key) for key in sensor_target_map})
    rows: list[dict[str, str]] = []
    for sensor in sensors:
        motifs = sensor_target_map.get(sensor, ())
        rows.append(
            {
                "Sensor": sensor,
                "Relevant stress": str(relevant_stress_map.get(sensor, "not declared")),
                "Relevant motifs": ", ".join(motifs) if motifs else "not declared",
            }
        )
    return rows


def retron_source_selector_rows(bundle: RetronReviewBundle) -> list[dict[str, str | int]]:
    counts: dict[str, int] = {}
    for source in bundle.sources:
        label = str(source.label)
        counts[label] = counts.get(label, 0) + 1
    return [
        _source_selector_row(source=source, idx=idx, duplicate_counts=counts)
        for idx, source in enumerate(bundle.sources)
    ]


def source_rows(bundle: RetronReviewBundle) -> list[dict[str, str]]:
    return [
        {
            "Label": source.label,
            "Experiment": source.experiment_id,
            "Config": str(source.config_path) if source.config_path is not None else "manifest-only",
            "Summary export": str(source.summary_path),
            "Trace export": str(source.trace_path),
        }
        for source in bundle.sources
    ]


def retron_source_surface_overview_rows(
    source: RetronReviewSource,
    surface: RetronReviewSourceSurface,
) -> list[dict[str, str]]:
    return [
        {"Field": "Source label", "Value": source.label},
        {"Field": "Experiment", "Value": surface.experiment_title or source.experiment_id},
        {"Field": "Protocol", "Value": surface.protocol_id},
        {"Field": "Compiled plots", "Value": str(len(surface.plot_catalog_rows))},
        {"Field": "Dataframe records", "Value": str(len(surface.record_paths))},
    ]


def _source_selector_row(
    *,
    source: RetronReviewSource,
    idx: int,
    duplicate_counts: Mapping[str, int],
) -> dict[str, str | int]:
    presentation = _source_selector_presentation(source)
    return {
        "Selector label": presentation.selector_label(
            experiment_id=str(source.experiment_id),
            duplicate_count=duplicate_counts.get(str(source.label), 0),
        ),
        "Index": idx,
    }


def _source_selector_presentation(source: RetronReviewSource) -> retron_review_catalog.SourceSelectorPresentation:
    return retron_review_catalog.source_selector_presentation(
        label=str(source.label),
        experiment_id=str(source.experiment_id),
    )


def retron_figure_option_rows(figures: list[Any] | tuple[Any, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for figure in figures:
        filename = str(getattr(figure, "filename", ""))
        rows.append(
            {
                "Filename": filename,
                "Label": retron_figure_label(filename),
            }
        )
    return rows


def render_retron_experiment_plot(
    plot_spec: Mapping[str, Any],
    *,
    datasets: Mapping[str, pd.DataFrame],
) -> RetronNotebookPlotResult:
    payload = retron_review_experiment_plots.render_experiment_plot_payload(plot_spec, datasets=datasets)
    metadata = _experiment_plot_guide(payload.plot_id)

    styled_figures = tuple(_prepare_notebook_plot_figure(item) for item in payload.figures)
    return RetronNotebookPlotResult(
        plot_id=payload.plot_id,
        title=str(payload.with_cfg.get("title") or metadata.title),
        stage=metadata.stage,
        question=metadata.question,
        math=metadata.math,
        meaning=metadata.meaning,
        source_record=metadata.record,
        figures=styled_figures,
        supporting_table=payload.supporting_table.reset_index(drop=True),
        supporting_table_title=payload.supporting_table_title,
    )


def render_retron_source_plot(
    source: RetronReviewSource,
    *,
    plot_id: str,
) -> RetronNotebookPlotResult:
    surface = load_retron_source_surface(source)
    plot_spec = retron_review_experiment_plots.source_plot_spec(surface=surface, plot_id=plot_id, label=source.label)
    semantic_record_ids = {
        "semantic_metrics/summary",
        "semantic_metrics/trace",
    } & set(
        retron_review_experiment_plots.source_plot_record_ids(
            plot_spec,
            record_paths=dict(surface.record_paths),
        )
    )
    semantic_datasets = (
        load_retron_source_semantic_datasets(
            source,
            record_ids=tuple(sorted(semantic_record_ids)),
        )
        if semantic_record_ids
        else {}
    )
    datasets = retron_review_experiment_plots.load_source_plot_datasets(
        surface=surface,
        plot_spec=plot_spec,
        load_frame=lambda record_id, path: load_retron_source_record_frame(
            source,
            record_id=record_id,
            path=path,
        ),
        semantic_datasets=semantic_datasets,
    )
    return render_retron_experiment_plot(plot_spec, datasets=datasets)


def render_retron_aggregate_plot(
    plot_id: str,
    *,
    summary_df: pd.DataFrame,
    sensor_target_map: Mapping[str, tuple[str, ...]],
    score_metric: str,
    architecture_x: str,
    expected_mode: str,
    fingerprint_sponge: str | None,
) -> RetronAggregatePlotResult:
    selected_plot_id = str(plot_id)
    render = retron_review_aggregate_plots.render_aggregate_plot_payload(
        selected_plot_id,
        summary_df=summary_df,
        sensor_target_map=sensor_target_map,
        score_metric=score_metric,
        architecture_x=architecture_x,
        expected_mode=expected_mode,
        fingerprint_sponge=fingerprint_sponge,
    )

    return RetronAggregatePlotResult(
        plot_id=selected_plot_id,
        title=render.guide.title,
        question=render.guide.question,
        math=render.guide.math,
        meaning=render.guide.meaning,
        figure=_style_notebook_figure(render.payload.figure) if render.payload.figure is not None else None,
        supporting_table=render.payload.supporting_table.reset_index(drop=True),
        supporting_table_title=render.supporting_table_title,
    )


def _summary_plot_config(
    *,
    plot_spec: Mapping[str, Any],
    with_cfg: Mapping[str, Any],
) -> retron_review_experiment_plots.SummaryPlotConfig:
    return retron_review_experiment_plots.summary_plot_config(plot_spec=plot_spec, with_cfg=with_cfg)


def _summary_supporting_table(
    summary_df: pd.DataFrame,
    *,
    view: str,
    metric: str,
    burden_metric: str,
) -> pd.DataFrame:
    return retron_review_experiment_plots.summary_supporting_table(
        summary_df,
        view=view,
        metric=metric,
        burden_metric=burden_metric,
    )


def try_render_retron_aggregate_plot(
    plot_id: str,
    *,
    summary_df: pd.DataFrame,
    sensor_target_map: Mapping[str, tuple[str, ...]],
    score_metric: str,
    architecture_x: str,
    expected_mode: str,
    fingerprint_sponge: str | None,
) -> tuple[RetronAggregatePlotResult | None, str | None]:
    try:
        return (
            render_retron_aggregate_plot(
                plot_id,
                summary_df=summary_df,
                sensor_target_map=sensor_target_map,
                score_metric=score_metric,
                architecture_x=architecture_x,
                expected_mode=expected_mode,
                fingerprint_sponge=fingerprint_sponge,
            ),
            None,
        )
    except Exception as exc:
        return None, str(exc)


def _experiment_plot_guide(plot_id: str) -> retron_review_catalog.ExperimentPlotGuideMetadata:
    return retron_review_catalog.experiment_plot_guide(str(plot_id))


def _plot_stage_rank(stage: str) -> int:
    try:
        return retron_review_catalog.PLOT_STAGE_ORDER.index(stage)
    except ValueError:
        return len(retron_review_catalog.PLOT_STAGE_ORDER)
