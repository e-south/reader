from __future__ import annotations

import pytest

from reader.errors import ConfigError
from reader.protocols import ProtocolBinding, builtin_protocol_catalog


@pytest.mark.parametrize(
    ("inputs", "analysis", "legacy_key"),
    [
        ({"reference": {"stat": "mean"}}, {}, "stat"),
        ({}, {"logic_symmetry": {"replicate_stat": "mean"}}, "replicate_stat"),
        ({"fold_change": {"report_times": [8.0], "agg": "median"}}, {}, "agg"),
    ],
)
def test_logic_sfxi_screen_rejects_legacy_observation_aggregation_keys(
    inputs: dict,
    analysis: dict,
    legacy_key: str,
) -> None:
    with pytest.raises(ConfigError, match=legacy_key):
        builtin_protocol_catalog().bind(
            ProtocolBinding(
                id="logic/sfxi_screen",
                inputs=inputs,
                analysis=analysis,
            )
        )


def test_logic_sfxi_screen_accepts_observation_stat_keys() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="logic/sfxi_screen",
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
    ("section", "legacy_key", "value"),
    [
        ("aggregation", "replicate_stat", "mean"),
        ("aggregation", "bootstrap_samples", 100),
        ("aggregation", "confidence_level", 0.95),
        ("quality", "min_replicates_per_state", 2),
    ],
)
def test_response_window_compile_rejects_retired_observation_keys(
    section: str,
    legacy_key: str,
    value: object,
) -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/response_window",
            analysis={section: {legacy_key: value}},
        )
    )

    with pytest.raises(ConfigError, match=legacy_key):
        protocol.compile()
