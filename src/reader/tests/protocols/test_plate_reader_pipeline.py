from __future__ import annotations

from reader.protocols.compilers.plate_reader_pipeline import (
    DUAL_REPORTER_BASE_RECIPE_ID,
    SINGLE_REPORTER_BASE_RECIPE_ID,
    SYNERGY_H1_INGEST_RECIPE_ID,
    compose_dual_reporter_pipeline,
    compose_single_reporter_pipeline,
)


def test_dual_reporter_pipeline_preserves_steps_configuration_and_provenance() -> None:
    steps = compose_dual_reporter_pipeline(
        ingest_channels=["OD600", "CFP", "YFP"],
        blank_config={"stat": "median"},
        overflow_config={"mode": "clip"},
    )

    assert [step.id for step in steps] == [
        "ingest",
        "merge_map",
        "labels",
        "blank",
        "overflow",
        "ratio_yfp_cfp",
        "ratio_cfp_od600",
        "ratio_yfp_od600",
    ]
    assert steps[0].with_ == {"channels": ["OD600", "CFP", "YFP"]}
    assert steps[0].source_recipe is not None
    assert steps[0].source_recipe.recipe == SYNERGY_H1_INGEST_RECIPE_ID
    assert steps[3].with_ == {"stat": "median"}
    assert steps[4].with_ == {"mode": "clip"}
    assert all(step.source_recipe is not None for step in steps)
    assert {step.source_recipe.recipe for step in steps[1:] if step.source_recipe} == {DUAL_REPORTER_BASE_RECIPE_ID}


def test_single_reporter_pipeline_binds_channels_and_recipe_arguments() -> None:
    steps = compose_single_reporter_pipeline(
        ingest_channels=["OD700", "mCherry"],
        reporter_channel="mCherry",
        normalizer_channel="OD700",
        blank_config={},
        overflow_config={},
    )

    ratio = steps[-2]
    assert ratio.id == "ratio_reporter_normalizer"
    assert ratio.with_ == {
        "name": "mCherry/OD700",
        "numerator": "mCherry",
        "denominator": "OD700",
    }
    assert ratio.source_recipe is not None
    assert ratio.source_recipe.recipe == SINGLE_REPORTER_BASE_RECIPE_ID
    assert ratio.source_recipe.with_ == {
        "reporter_channel": "mCherry",
        "normalizer_channel": "OD700",
    }
    sample_measurements = steps[-1]
    assert sample_measurements.id == "sample_measurements"
    assert sample_measurements.plugin == "validator/to_tidy_plus_map"
    assert sample_measurements.reads["df"].record_id == "ratio_reporter_normalizer/df"
    assert sample_measurements.writes["df"].record_id == "sample_measurements/df"
    assert sample_measurements.with_ == {
        "include_types": ["SAMPLE"],
        "require_columns": ["treatment", "design_id"],
        "require_non_null": True,
        "trim_and_require_non_blank": ["treatment", "design_id"],
        "require_finite": ["time", "value"],
    }
    assert {step.source_recipe.recipe for step in steps[1:] if step.source_recipe is not None} == {
        SINGLE_REPORTER_BASE_RECIPE_ID
    }
