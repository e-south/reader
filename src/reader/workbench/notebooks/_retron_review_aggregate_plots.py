from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from reader.workbench.notebooks import _retron_review_aggregate_data as retron_review_aggregate_data
from reader.workbench.notebooks import _retron_review_aggregate_figures as retron_review_aggregate_figures
from reader.workbench.notebooks import _retron_review_catalog as retron_review_catalog

aggregate_on_target_scores = retron_review_aggregate_data.aggregate_on_target_scores
available_multifunctional_sponges = retron_review_aggregate_data.available_multifunctional_sponges
build_aggregate_pareto_frame = retron_review_aggregate_data.build_aggregate_pareto_frame
build_architecture_frame = retron_review_aggregate_data.build_architecture_frame
build_expected_vs_observed_frame = retron_review_aggregate_data.build_expected_vs_observed_frame
build_fingerprint_frame = retron_review_aggregate_data.build_fingerprint_frame
build_specificity_matrix = retron_review_aggregate_data.build_specificity_matrix


@dataclass(frozen=True)
class AggregatePlotPayload:
    figure: Any | None
    supporting_table: pd.DataFrame


@dataclass(frozen=True)
class AggregatePlotRenderResult:
    guide: retron_review_catalog.AggregatePlotGuideMetadata
    supporting_table_title: str
    payload: AggregatePlotPayload


@dataclass(frozen=True)
class _AggregatePlotContext:
    summary_df: pd.DataFrame
    sensor_target_map: Mapping[str, tuple[str, ...]]
    score_metric: str
    architecture_x: str
    expected_mode: str
    fingerprint_sponge: str | None


@dataclass(frozen=True)
class _AggregatePlotSpec:
    guide: retron_review_catalog.AggregatePlotGuideMetadata
    payload_builder: Callable[[_AggregatePlotContext], AggregatePlotPayload]
    supporting_table_title: str


def render_aggregate_plot_payload(
    plot_id: str,
    *,
    summary_df: pd.DataFrame,
    sensor_target_map: Mapping[str, tuple[str, ...]],
    score_metric: str,
    architecture_x: str,
    expected_mode: str,
    fingerprint_sponge: str | None,
) -> AggregatePlotRenderResult:
    spec = aggregate_plot_spec(plot_id)
    context = _AggregatePlotContext(
        summary_df=summary_df,
        sensor_target_map=sensor_target_map,
        score_metric=score_metric,
        architecture_x=architecture_x,
        expected_mode=expected_mode,
        fingerprint_sponge=fingerprint_sponge,
    )
    return AggregatePlotRenderResult(
        guide=spec.guide,
        supporting_table_title=spec.supporting_table_title,
        payload=spec.payload_builder(context),
    )


def aggregate_plot_spec(plot_id: str) -> _AggregatePlotSpec:
    try:
        return aggregate_plot_specs()[str(plot_id)]
    except KeyError as exc:
        raise ValueError(f"retron_review: unknown aggregate plot id {plot_id!r}") from exc


def aggregate_plot_specs() -> dict[str, _AggregatePlotSpec]:
    payload_builders: dict[str, Callable[[_AggregatePlotContext], AggregatePlotPayload]] = {
        "specificity_matrix": _specificity_matrix_payload,
        "pareto_ranking": _aggregate_pareto_payload,
        "architecture_plot": _architecture_plot_payload,
        "expected_vs_observed": _expected_vs_observed_payload,
        "sponge_fingerprint": _sponge_fingerprint_payload,
    }
    catalog_entries = retron_review_catalog.RETRON_AGGREGATE_PLOT_CATALOG
    missing = sorted(set(catalog_entries) - set(payload_builders))
    extra = sorted(set(payload_builders) - set(catalog_entries))
    if missing or extra:
        raise RuntimeError(
            f"retron_review: aggregate plot catalog and payload builders diverged (missing={missing}, extra={extra})"
        )
    return {
        plot_id: _AggregatePlotSpec(
            guide=entry.guide,
            payload_builder=payload_builders[plot_id],
            supporting_table_title=entry.supporting_table_title,
        )
        for plot_id, entry in catalog_entries.items()
    }


def _specificity_matrix_payload(context: _AggregatePlotContext) -> AggregatePlotPayload:
    matrix = build_specificity_matrix(context.summary_df, score_metric=context.score_metric)
    supporting_table = matrix.reset_index().rename(columns={"index": "sponge"})
    return AggregatePlotPayload(
        figure=retron_review_aggregate_figures.build_specificity_matrix_figure(
            matrix=matrix,
            score_metric=context.score_metric,
        ),
        supporting_table=supporting_table,
    )


def _aggregate_pareto_payload(context: _AggregatePlotContext) -> AggregatePlotPayload:
    supporting_table = build_aggregate_pareto_frame(context.summary_df, score_metric=context.score_metric)
    return AggregatePlotPayload(
        figure=retron_review_aggregate_figures.build_aggregate_pareto_figure(
            pareto_df=supporting_table,
            score_metric=context.score_metric,
            burden_metric=retron_review_catalog.DEFAULT_RETRON_BURDEN_METRIC,
        ),
        supporting_table=supporting_table,
    )


def _architecture_plot_payload(context: _AggregatePlotContext) -> AggregatePlotPayload:
    supporting_table = build_architecture_frame(
        context.summary_df,
        sensor_target_map=dict(context.sensor_target_map),
        score_metric=context.score_metric,
    )
    return AggregatePlotPayload(
        figure=retron_review_aggregate_figures.build_architecture_figure(
            architecture_df=supporting_table,
            score_metric=context.score_metric,
            architecture_x=context.architecture_x,
        ),
        supporting_table=supporting_table,
    )


def _expected_vs_observed_payload(context: _AggregatePlotContext) -> AggregatePlotPayload:
    supporting_table = build_expected_vs_observed_frame(
        context.summary_df,
        sensor_target_map=dict(context.sensor_target_map),
        score_metric=context.score_metric,
    )
    return AggregatePlotPayload(
        figure=retron_review_aggregate_figures.build_expected_vs_observed_figure(
            expected_vs_observed_df=supporting_table,
            score_metric=context.score_metric,
            expected_mode=context.expected_mode,
        ),
        supporting_table=supporting_table,
    )


def _sponge_fingerprint_payload(context: _AggregatePlotContext) -> AggregatePlotPayload:
    supporting_table = build_fingerprint_frame(
        context.summary_df,
        score_metric=context.score_metric,
        fingerprint_sponge=context.fingerprint_sponge,
    )
    return AggregatePlotPayload(
        figure=retron_review_aggregate_figures.build_fingerprint_figure(
            fingerprint_df=supporting_table,
            score_metric=context.score_metric,
        ),
        supporting_table=supporting_table,
    )
