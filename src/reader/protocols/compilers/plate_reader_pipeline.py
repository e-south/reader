from __future__ import annotations

from typing import Any

from reader.workbench.decl.model import PluginStepDecl, RecordInputDecl, RecordOutputDecl, ResourceInputDecl

from .common import _deep_merge, _step

SYNERGY_H1_INGEST_RECIPE_ID = "plate_reader/synergy_h1"
DUAL_REPORTER_BASE_RECIPE_ID = "plate_reader/dual_reporter_screen_base"
SINGLE_REPORTER_BASE_RECIPE_ID = "plate_reader/single_reporter_screen_base"


def compose_dual_reporter_pipeline(
    *,
    ingest_channels: list[str],
    blank_config: dict[str, Any],
    overflow_config: dict[str, Any],
) -> tuple[PluginStepDecl, ...]:
    return (
        _step(
            id="ingest",
            plugin="ingest/synergy_h1",
            with_={"channels": ingest_channels},
            source_recipe=SYNERGY_H1_INGEST_RECIPE_ID,
        ),
        *_sample_map_and_preprocessing_steps(
            blank_config=blank_config,
            overflow_config=overflow_config,
            source_recipe=DUAL_REPORTER_BASE_RECIPE_ID,
        ),
        _step(
            id="ratio_yfp_cfp",
            plugin="transform/ratio",
            reads={"df": RecordInputDecl(record_id="overflow/df")},
            with_={"name": "YFP/CFP", "numerator": "YFP", "denominator": "CFP"},
            source_recipe=DUAL_REPORTER_BASE_RECIPE_ID,
        ),
        _step(
            id="ratio_cfp_od600",
            plugin="transform/ratio",
            reads={"df": RecordInputDecl(record_id="ratio_yfp_cfp/df")},
            with_={"name": "CFP/OD600", "numerator": "CFP", "denominator": "OD600"},
            source_recipe=DUAL_REPORTER_BASE_RECIPE_ID,
        ),
        _step(
            id="ratio_yfp_od600",
            plugin="transform/ratio",
            reads={"df": RecordInputDecl(record_id="ratio_cfp_od600/df")},
            with_={"name": "YFP/OD600", "numerator": "YFP", "denominator": "OD600"},
            source_recipe=DUAL_REPORTER_BASE_RECIPE_ID,
        ),
    )


def compose_single_reporter_pipeline(
    *,
    ingest_channels: list[str],
    reporter_channel: str,
    normalizer_channel: str,
    blank_config: dict[str, Any],
    overflow_config: dict[str, Any],
) -> tuple[PluginStepDecl, ...]:
    ratio_name = f"{reporter_channel}/{normalizer_channel}"
    recipe_arguments = {
        "reporter_channel": reporter_channel,
        "normalizer_channel": normalizer_channel,
    }
    return (
        _step(
            id="ingest",
            plugin="ingest/synergy_h1",
            with_={"channels": ingest_channels},
            source_recipe=SYNERGY_H1_INGEST_RECIPE_ID,
        ),
        *_sample_map_and_preprocessing_steps(
            blank_config=blank_config,
            overflow_config=overflow_config,
            source_recipe=SINGLE_REPORTER_BASE_RECIPE_ID,
            recipe_arguments=recipe_arguments,
        ),
        _step(
            id="ratio_reporter_normalizer",
            plugin="transform/ratio",
            reads={"df": RecordInputDecl(record_id="overflow/df")},
            with_={
                "name": ratio_name,
                "numerator": reporter_channel,
                "denominator": normalizer_channel,
            },
            source_recipe=SINGLE_REPORTER_BASE_RECIPE_ID,
            source_recipe_with=recipe_arguments,
        ),
        _step(
            id="sample_measurements",
            plugin="validator/to_tidy_plus_map",
            reads={"df": RecordInputDecl(record_id="ratio_reporter_normalizer/df")},
            with_={
                "include_types": ["SAMPLE"],
                "require_columns": ["treatment", "design_id"],
                "require_non_null": True,
                "trim_and_require_non_blank": ["treatment", "design_id"],
                "require_finite": ["time", "value"],
            },
            writes={"df": RecordOutputDecl(record_id="sample_measurements/df")},
            source_recipe=SINGLE_REPORTER_BASE_RECIPE_ID,
            source_recipe_with=recipe_arguments,
        ),
    )


def _sample_map_and_preprocessing_steps(
    *,
    blank_config: dict[str, Any],
    overflow_config: dict[str, Any],
    source_recipe: str,
    recipe_arguments: dict[str, Any] | None = None,
) -> tuple[PluginStepDecl, ...]:
    source_recipe_with = dict(recipe_arguments or {})
    return (
        _step(
            id="merge_map",
            plugin="transform/sample_map",
            reads={
                "df": RecordInputDecl(record_id="ingest/df"),
                "sample_map": ResourceInputDecl(resource_id="sample_map"),
            },
            source_recipe=source_recipe,
            source_recipe_with=source_recipe_with,
        ),
        _step(
            id="labels",
            plugin="transform/assay_labels",
            reads={"df": RecordInputDecl(record_id="merge_map/df")},
            source_recipe=source_recipe,
            source_recipe_with=source_recipe_with,
        ),
        _step(
            id="blank",
            plugin="transform/blank_correction",
            reads={"df": RecordInputDecl(record_id="labels/df")},
            with_=_deep_merge(blank_config),
            source_recipe=source_recipe,
            source_recipe_with=source_recipe_with,
        ),
        _step(
            id="overflow",
            plugin="transform/overflow_handling",
            reads={"df": RecordInputDecl(record_id="blank/df")},
            with_=_deep_merge(overflow_config),
            source_recipe=source_recipe,
            source_recipe_with=source_recipe_with,
        ),
    )
