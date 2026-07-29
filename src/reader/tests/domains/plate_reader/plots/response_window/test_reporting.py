from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reader.domains.plate_reader.plots.response_window.reporting import (
    _spearman_rank_correlation,
)
from reader.domains.plate_reader.plots.response_window.reporting_quality_plots import (
    write_repeat_plot,
)


def test_spearman_rank_correlation_uses_average_ranks_for_ties() -> None:
    correlation = _spearman_rank_correlation(
        np.asarray([1.0, 1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.0, 2.0, 3.0]),
    )

    assert correlation == pytest.approx(5.0 / 6.0)


def test_spearman_rank_correlation_handles_constant_series_explicitly() -> None:
    constant = np.asarray([2.0, 2.0, 2.0])

    assert _spearman_rank_correlation(constant, constant) == 1.0
    assert np.isnan(_spearman_rank_correlation(constant, np.asarray([1.0, 2.0, 3.0])))


def test_repeat_plot_renders_empty_state_when_no_design_repeats(tmp_path: Path) -> None:
    (tmp_path / "plots").mkdir()
    manifest_row = write_repeat_plot(
        pd.DataFrame(),
        display={
            "channels": {
                "response_ratio": "YFP/CFP",
                "magnitude_ratio": "YFP/OD600",
                "growth": "OD600",
                "reference_design_id": "pDual-10",
            }
        },
        out_dir=tmp_path,
    )

    assert manifest_row["plot_id"] == "repeated_design_agreement"
    assert (tmp_path / manifest_row["path"]).is_file()
