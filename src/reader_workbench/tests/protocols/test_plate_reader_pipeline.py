from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from reader_workbench.plugins.transform.overflow import OverflowCfg, OverflowHandling
from reader_workbench.plugins.validator.to_tidy_plus_map import PromoteCfg, PromoteToTidyPlusMap
from reader_workbench.protocols.compilers.plate_reader_pipeline import (
    DUAL_REPORTER_BASE_RECIPE_ID,
    GROWTH_BASE_RECIPE_ID,
    SINGLE_REPORTER_BASE_RECIPE_ID,
    SYNERGY_H1_INGEST_RECIPE_ID,
    compose_dual_reporter_pipeline,
    compose_growth_pipeline,
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


def test_growth_pipeline_preserves_one_channel_without_reporter_semantics() -> None:
    steps = compose_growth_pipeline(
        ingest_channels=["OD700"],
        growth_channel="OD700",
        blank_config={"stat": "median"},
        overflow_config={"mode": "clip"},
    )

    assert [step.id for step in steps] == [
        "ingest",
        "merge_map",
        "labels",
        "blank",
        "overflow",
        "sample_measurements",
    ]
    assert steps[0].with_ == {"channels": ["OD700"]}
    sample_measurements = steps[-1]
    assert sample_measurements.plugin == "validator/to_tidy_plus_map"
    assert sample_measurements.reads["df"].record_id == "overflow/df"
    assert sample_measurements.writes["df"].record_id == "sample_measurements/df"
    assert all(
        step.source_recipe is not None and step.source_recipe.recipe == GROWTH_BASE_RECIPE_ID for step in steps[1:]
    )
    assert all(
        step.source_recipe is not None and step.source_recipe.with_ == {"growth_channel": "OD700"} for step in steps[1:]
    )


def test_growth_pipeline_promotes_classified_instrument_overflow() -> None:
    steps = compose_growth_pipeline(
        ingest_channels=["OD700"],
        growth_channel="OD700",
        blank_config={},
        overflow_config={"action": "none"},
    )
    overflow = next(step for step in steps if step.id == "overflow")
    promotion = next(step for step in steps if step.id == "sample_measurements")
    frame = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["OD700"],
            "value": [float("inf")],
            "type": ["SAMPLE"],
            "treatment": ["condition-a"],
            "design_id": ["design-a"],
            "overflow": [False],
        }
    )
    context = SimpleNamespace(logger=None)

    classified = OverflowHandling().run(
        context,
        {"df": frame},
        OverflowCfg.model_validate(overflow.with_),
    )["df"]
    promoted = PromoteToTidyPlusMap().run(
        context,
        {"df": classified},
        PromoteCfg.model_validate(promotion.with_),
    )["df"]

    assert promoted["value"].tolist() == [float("inf")]
    assert promoted["value_instrument_overflow"].tolist() == [True]
    assert promoted["value_bound_kind"].tolist() == ["lower"]


def test_growth_pipeline_drop_policy_promotes_only_exact_retained_rows() -> None:
    steps = compose_growth_pipeline(
        ingest_channels=["OD700"],
        growth_channel="OD700",
        blank_config={},
        overflow_config={"action": "drop"},
    )
    overflow = next(step for step in steps if step.id == "overflow")
    promotion = next(step for step in steps if step.id == "sample_measurements")
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2", "A3"],
            "time": [0.0, 0.0, 0.0],
            "channel": ["OD700", "OD700", "OD700"],
            "value": [0.1, float("inf"), 0.3],
            "type": ["SAMPLE", "SAMPLE", "SAMPLE"],
            "treatment": ["condition-a", "condition-a", "condition-a"],
            "design_id": ["design-a", "design-a", "design-a"],
            "overflow": [False, False, True],
        }
    )
    context = SimpleNamespace(logger=None)

    classified = OverflowHandling().run(
        context,
        {"df": frame},
        OverflowCfg.model_validate(overflow.with_),
    )["df"]
    promoted = PromoteToTidyPlusMap().run(
        context,
        {"df": classified},
        PromoteCfg.model_validate(promotion.with_),
    )["df"]

    assert promoted["position"].tolist() == ["A1"]
    assert promoted["value"].tolist() == [0.1]
    assert promoted["value_policy_clipped"].tolist() == [False]
    assert promoted["value_instrument_overflow"].tolist() == [False]
    assert promoted["value_bound_kind"].tolist() == ["exact"]
