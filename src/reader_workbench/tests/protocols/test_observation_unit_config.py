from __future__ import annotations

import pytest

from reader_workbench.errors import ConfigError
from reader_workbench.protocols import ProtocolBinding, builtin_protocol_catalog


@pytest.mark.parametrize(
    ("inputs", "analysis", "legacy_key"),
    [
        ({"reference": {"stat": "mean"}}, {}, "stat"),
        ({}, {"logic_symmetry": {"replicate_stat": "mean"}}, "replicate_stat"),
        ({"fold_change": {"report_times": [8.0], "agg": "median"}}, {}, "agg"),
    ],
)
def test_logic_four_state_vector_screen_rejects_legacy_observation_aggregation_keys(
    inputs: dict,
    analysis: dict,
    legacy_key: str,
) -> None:
    with pytest.raises(ConfigError, match=legacy_key):
        builtin_protocol_catalog().bind(
            ProtocolBinding(
                id="logic/four_state_vector_screen",
                inputs=inputs,
                analysis=analysis,
            )
        )


def test_logic_four_state_vector_screen_accepts_observation_stat_keys() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="logic/four_state_vector_screen",
            inputs={
                "reference": {"design_id": "REF", "observation_stat": "mean"},
                "fold_change": {"report_times": [8.0], "observation_stat": "median"},
            },
            analysis={"logic_symmetry": {"observation_stat": "mean"}},
        )
    )

    assert protocol.effective_inputs()["reference"]["observation_stat"] == "mean"
    assert protocol.effective_inputs()["fold_change"]["observation_stat"] == "median"
    assert protocol.effective_analysis()["logic_symmetry"]["observation_stat"] == "mean"


@pytest.mark.parametrize(
    ("protocol_id", "fold_change_step_id"),
    [
        ("plate_reader/dual_reporter_screen", "fold_change__yfp_over_cfp"),
        ("plate_reader/single_reporter_screen", "fold_change__single_reporter"),
    ],
)
def test_plate_reader_fold_change_binds_expected_treatments_into_compiled_step(
    protocol_id: str,
    fold_change_step_id: str,
) -> None:
    expected_treatments = ["baseline", "induced"]
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id=protocol_id,
            inputs={
                "fold_change": {
                    "report_times": [8.0],
                    "expected_treatments": expected_treatments,
                }
            },
            analysis={"include_fold_change": True},
        )
    )

    step = next(item for item in protocol.compile().pipeline if item.id == fold_change_step_id)
    effective_step_config = protocol.effective_plugin_config(plugin_id=step.plugin, step_with=step.with_)

    assert protocol.effective_inputs()["fold_change"]["expected_treatments"] == expected_treatments
    assert effective_step_config["expected_treatments"] == expected_treatments


@pytest.mark.parametrize(
    ("section", "legacy_key", "value"),
    [
        ("aggregation", "replicate_stat", "mean"),
        ("aggregation", "bootstrap_samples", 100),
        ("aggregation", "confidence_level", 0.95),
        ("quality", "min_replicates_per_state", 2),
    ],
)
def test_four_state_event_window_compile_rejects_retired_observation_keys(
    section: str,
    legacy_key: str,
    value: object,
) -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/four_state_event_window",
            analysis={section: {legacy_key: value}},
        )
    )

    with pytest.raises(ConfigError, match=legacy_key):
        protocol.compile()
