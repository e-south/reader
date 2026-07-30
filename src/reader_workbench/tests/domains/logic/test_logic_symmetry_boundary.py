from __future__ import annotations

from inspect import signature

import matplotlib.pyplot as plt
import pandas as pd

from reader_workbench.domains.logic.logic_symmetry import render_logic_symmetry, summarize_logic_symmetry


def logic_symmetry_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["A1", "A2", "A3", "A4"],
            "time": [1.0, 1.0, 1.0, 1.0],
            "channel": ["signal", "signal", "signal", "signal"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "design_id": ["design_a", "design_a", "design_a", "design_a"],
            "batch": [0, 0, 0, 0],
            "condition": ["off", "input_a", "input_b", "both"],
        }
    )


def test_logic_symmetry_domain_separates_summary_from_rendering() -> None:
    assert "output_dir" not in signature(summarize_logic_symmetry).parameters
    assert "output_dir" not in signature(render_logic_symmetry).parameters

    table = summarize_logic_symmetry(
        logic_symmetry_input(),
        response_channel="signal",
        design_by=["design_id"],
        treatment_column="condition",
        treatment_map={"00": "off", "10": "input_a", "01": "input_b", "11": "both"},
    )
    figure = render_logic_symmetry(table, figsize=(4, 3), dpi=72)

    assert table["design_id"].tolist() == ["design_a"]
    assert figure is not None
    plt.close(figure)
