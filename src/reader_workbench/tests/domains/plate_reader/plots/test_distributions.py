from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from reader_workbench.domains.plate_reader.plots.distributions import plot_distributions


def _plot(frame: pd.DataFrame):
    return plot_distributions(
        df=frame,
        blanks=frame.iloc[0:0],
        channels=["signal"],
        group_on="design_id",
        fig_kwargs={},
    )


def test_distributions_exclude_anonymous_partition_identities() -> None:
    frame = pd.DataFrame(
        {
            "channel": ["signal"] * 6,
            "value": [1.0, 1.1, 1.2, 5.0, 6.0, 7.0],
            "design_id": ["design_a", "design_a", "design_a", None, "nan", "  "],
        }
    )

    figures = _plot(frame)

    assert [figure.filename for figure in figures] == ["distrib__design_a"]
    for figure in figures:
        plt.close(figure.fig)


def test_distributions_emit_no_artifact_for_an_entirely_anonymous_partition() -> None:
    frame = pd.DataFrame(
        {
            "channel": ["signal"] * 3,
            "value": [5.0, 6.0, 7.0],
            "design_id": [None, "none", "  "],
        }
    )

    assert _plot(frame) == []


def test_distributions_filter_and_report_nonfinite_measurements(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    observed: list[np.ndarray] = []

    def capture_kdeplot(*, data: pd.DataFrame, x: str, **_: object) -> None:
        observed.append(data[x].to_numpy(dtype=float, copy=True))

    monkeypatch.setattr(
        "reader_workbench.domains.plate_reader.plots.distributions.sns.kdeplot",
        capture_kdeplot,
    )
    frame = pd.DataFrame(
        {
            "channel": ["signal"] * 5,
            "value": [1.0, 2.0, 3.0, np.inf, 100.0],
            "design_id": ["design_a"] * 5,
            "value_policy_clipped": [False, False, False, False, True],
            "value_instrument_overflow": [False, False, False, True, False],
            "value_bound_kind": ["exact", "exact", "exact", "lower", "lower"],
        }
    )

    with caplog.at_level("WARNING", logger="reader"):
        figures = _plot(frame)

    assert len(figures) == 1
    assert figures[0].description == (
        "Selected measurements: 5 observed, 2 omitted as bounded or non-finite; "
        "bounded or non-finite rows were omitted before density estimation."
    )
    assert len(observed) == 1
    assert observed[0].tolist() == [1.0, 2.0, 3.0]
    assert "distributions: withheld 2 bounded or non-finite observation(s)" in caplog.text
    for figure in figures:
        plt.close(figure.fig)


def test_distributions_omit_and_report_fully_censored_measurements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pd.DataFrame(
        {
            "channel": ["signal"] * 3,
            "value": [np.inf, np.inf, np.inf],
            "design_id": ["design_a"] * 3,
        }
    )

    with caplog.at_level("WARNING", logger="reader"):
        figures = _plot(frame)

    assert figures == []
    assert "distributions: withheld 3 bounded or non-finite observation(s)" in caplog.text


def test_distribution_description_includes_selected_nonfinite_blank_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "reader_workbench.domains.plate_reader.plots.distributions.sns.kdeplot",
        lambda **_: None,
    )
    frame = pd.DataFrame(
        {
            "channel": ["signal"] * 3,
            "value": [1.0, 2.0, 3.0],
            "design_id": ["design_a"] * 3,
        }
    )
    blanks = pd.DataFrame(
        {
            "channel": ["signal", "other"],
            "value": [np.inf, np.inf],
        }
    )

    figures = plot_distributions(
        df=frame,
        blanks=blanks,
        channels=["signal"],
        group_on="design_id",
        fig_kwargs={},
    )

    assert figures[0].description == (
        "Selected measurements: 4 observed, 1 omitted as bounded or non-finite; "
        "bounded or non-finite rows were omitted before density estimation."
    )
    plt.close(figures[0].fig)
