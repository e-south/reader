from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from reader.plugins.plot.logic_symmetry import LogicSymCfg, LogicSymmetryPlot
from reader.workbench.experiment import (
    AnnotationSemantics,
    OrderedStateSpaces,
    OrderedStateSpaceSpec,
)


def test_logic_symmetry_rejects_noncanonical_sfxi_state_space() -> None:
    annotations = AnnotationSemantics(
        ordered_state_spaces=OrderedStateSpaces(
            by_id={
                "states": OrderedStateSpaceSpec(
                    column="treatment",
                    state_order=("00", "01", "10", "11"),
                    source_values={"00": "A", "01": "C", "10": "B", "11": "D"},
                )
            }
        )
    )
    ctx = SimpleNamespace(
        experiment=SimpleNamespace(annotations=annotations),
        palette_book=None,
    )
    cfg = LogicSymCfg(response_channel="YFP/CFP", state_map_ref="states")

    with pytest.raises(ValueError, match="SFXI state space must declare exactly 00, 10, 01, 11 in that order"):
        LogicSymmetryPlot().render(ctx, {"df": pd.DataFrame()}, cfg)
