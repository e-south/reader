from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from reader.plugins.transform.overflow import OverflowCfg, OverflowHandling


def test_overflow_max_distinguishes_policy_clipping_from_instrument_overflow() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2", "A3"],
            "time": [0.0, 0.0, 0.0],
            "channel": ["YFP", "YFP", "YFP"],
            "value": [10.0, 100.0, float("inf")],
            "overflow": [False, False, True],
        }
    )

    logger = Mock()
    result = OverflowHandling().run(
        SimpleNamespace(logger=logger),
        {"df": frame},
        OverflowCfg(
            action="max",
            cap_strategy="provided",
            per_channel_caps={"YFP": 50.0},
        ),
    )["df"]

    assert result["value"].tolist() == [10.0, 50.0, 50.0]
    assert result["value_policy_clipped"].tolist() == [False, True, False]
    assert result["value_instrument_overflow"].tolist() == [False, False, True]
    assert result["value_bound_kind"].tolist() == ["exact", "lower", "lower"]
    logger.info.assert_called_once()
    logged = logger.info.call_args.args
    assert logged[2:4] == (1, 1)
    assert logged[4] == {"YFP": 1}
    assert logged[5] == {"YFP": 1}


def test_overflow_rejects_nonboolean_instrument_flags() -> None:
    frame = pd.DataFrame({"channel": ["YFP"], "value": [10.0], "overflow": ["False"]})

    with pytest.raises(ValueError, match="booleans"):
        OverflowHandling().run(
            SimpleNamespace(logger=None),
            {"df": frame},
            OverflowCfg(action="max", cap_strategy="provided", per_channel_caps={"YFP": 50.0}),
        )


def test_overflow_rejects_unclassified_nonfinite_values() -> None:
    frame = pd.DataFrame({"channel": ["YFP", "YFP"], "value": [10.0, float("inf")], "overflow": [False, False]})

    with pytest.raises(ValueError, match="non-finite"):
        OverflowHandling().run(
            SimpleNamespace(logger=None),
            {"df": frame},
            OverflowCfg(
                action="max",
                cap_strategy="provided",
                per_channel_caps={"YFP": 50.0},
                treat_inf_as_overflow=False,
            ),
        )
