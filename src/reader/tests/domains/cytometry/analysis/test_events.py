from __future__ import annotations

import math

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from reader.domains.cytometry.analysis import (
    CytometryAnalysisError,
    EventFilters,
    GateSpec,
    ThresholdSpec,
    analyze_events,
    distinct_string_values_by_column,
    gate_defaults,
    prepare_event_table,
    prepare_plot_events,
    prepare_plot_payload,
    scan_event_table,
)


def _tidy_events() -> pl.DataFrame:
    events = {
        "s1": {
            "design_id": "d1",
            "treatment": "dose",
            "sample_label": "sample 1",
            "values": [
                (10.0, 20.0, 9.0, -1.0),
                (15.0, 25.0, 14.0, 10.0),
                (1000.0, 20.0, 900.0, 100.0),
            ],
        },
        "s2": {
            "design_id": "d2",
            "treatment": "dose",
            "sample_label": "sample 2",
            "values": [
                (12.0, 22.0, 12.0, 20.0),
                (18.0, 28.0, 2.0, 30.0),
                (30.0, 30.0, 27.0, 40.0),
            ],
        },
    }
    rows: list[dict[str, object]] = []
    for sample_id, sample in events.items():
        for event_index, values in enumerate(sample["values"]):
            for channel, value in zip(
                ("FSC-A", "SSC-A", "FSC-H", "mCherry-A"),
                values,
                strict=True,
            ):
                rows.append(
                    {
                        "event_index": event_index,
                        "channel": channel,
                        "value": value,
                        "sample_id": sample_id,
                        "design_id": sample["design_id"],
                        "treatment": sample["treatment"],
                        "sample_label": sample["sample_label"],
                        "unused": "not projected",
                    }
                )
    return pl.DataFrame(rows)


def _reference_prepare(frame: pl.DataFrame, *, design_id: str | None = None) -> pd.DataFrame:
    channels = ["FSC-A", "FSC-H", "SSC-A", "mCherry-A"]
    work = frame.filter(pl.col("channel").is_in(channels))
    if design_id is not None:
        work = work.filter(pl.col("design_id") == design_id)
    return (
        work.with_columns(pl.col("value").cast(pl.Float64))
        .pivot(
            values="value",
            index=["sample_id", "event_index", "treatment", "design_id", "sample_label"],
            on="channel",
            aggregate_function="first",
        )
        .to_pandas(use_pyarrow_extension_array=False)
    )


def _canonical_pandas(frame: pl.DataFrame | pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if isinstance(frame, pl.DataFrame):
        frame = frame.to_pandas(use_pyarrow_extension_array=False)
    return frame.loc[:, columns].sort_values(columns[:2]).reset_index(drop=True)


def test_prepare_event_table_matches_reference_reduction_and_projects_before_collect(tmp_path) -> None:
    path = tmp_path / "events.parquet"
    _tidy_events().write_parquet(path)

    source = scan_event_table(path)
    actual = prepare_event_table(
        source,
        channels=("FSC-A", "FSC-H", "SSC-A", "mCherry-A"),
        filters=EventFilters(design_id="d1"),
    )
    expected = _reference_prepare(_tidy_events(), design_id="d1")
    columns = [
        "sample_id",
        "event_index",
        "treatment",
        "design_id",
        "sample_label",
        "FSC-A",
        "FSC-H",
        "SSC-A",
        "mCherry-A",
    ]

    assert_frame_equal(
        _canonical_pandas(actual, columns),
        _canonical_pandas(expected, columns),
        check_dtype=False,
    )
    assert actual.get_column("event_index").to_list() == [0, 1, 2]
    assert "unused" not in actual.columns


def test_distinct_string_values_are_collected_in_one_column_map() -> None:
    values = distinct_string_values_by_column(
        _tidy_events().lazy(),
        ("channel", "design_id", "missing"),
    )

    assert values == {
        "channel": ["FSC-A", "FSC-H", "SSC-A", "mCherry-A"],
        "design_id": ["d1", "d2"],
        "missing": [],
    }


def test_prepare_event_table_fails_fast_when_a_selected_channel_is_absent() -> None:
    with pytest.raises(CytometryAnalysisError, match="Missing channels after pivot: absent"):
        prepare_event_table(
            _tidy_events().lazy(),
            channels=("FSC-A", "absent"),
        )


def test_prepare_event_table_rejects_duplicate_event_channel_rows() -> None:
    events = _tidy_events()
    duplicated = pl.concat((events, events.head(1)))

    with pytest.raises(
        CytometryAnalysisError,
        match=r"duplicate pivot key rows.*sample_id.*event_index.*channel",
    ):
        prepare_event_table(
            duplicated.lazy(),
            channels=("FSC-A", "FSC-H", "SSC-A", "mCherry-A"),
        )


def test_gate_and_summary_values_match_current_notebook_semantics() -> None:
    wide = prepare_event_table(
        _tidy_events(),
        channels=("FSC-A", "FSC-H", "SSC-A", "mCherry-A"),
    )
    result = analyze_events(
        wide,
        gate=GateSpec(
            cells_x_channel="FSC-A",
            cells_y_channel="SSC-A",
            cells_x_range=(0.0, 100.0),
            cells_y_range=(0.0, 100.0),
            singlet_x_channel="FSC-A",
            singlet_y_channel="FSC-H",
            singlet_ratio_range=(0.8, 1.2),
        ),
        threshold=ThresholdSpec(channel="mCherry-A", value=5.0),
    )

    assert result.gated_events.select("sample_id", "event_index").rows() == [
        ("s1", 0),
        ("s1", 1),
        ("s2", 0),
        ("s2", 2),
    ]

    counts = result.gate_counts_sample.sort("sample_id")
    assert counts.select("sample_id", "n_total_events", "n_cells_gate", "n_singlets").rows() == [
        ("s1", 3, 2, 2),
        ("s2", 3, 3, 2),
    ]
    assert counts.get_column("pct_cells").to_list() == [100.0 * 2.0 / 3.0, 100.0]
    assert counts.get_column("pct_singlets_of_cells").to_list() == [100.0, 100.0 * 2.0 / 3.0]
    assert counts.get_column("pct_final").to_list() == [100.0 * 2.0 / 3.0, 100.0 * 2.0 / 3.0]

    stats = result.stats_sample.sort("sample_id")
    assert stats.get_column("fluor_median").to_list() == [4.5, 30.0]
    assert stats.get_column("fluor_mean").to_list() == [4.5, 30.0]
    assert math.isclose(stats.get_column("fluor_geomean")[0], 10.0)
    assert math.isclose(stats.get_column("fluor_geomean")[1], math.sqrt(800.0))
    assert stats.get_column("fluor_p90").to_list() == [8.9, 38.0]
    assert stats.get_column("fluor_p99").to_list() == [9.89, 39.8]
    assert stats.get_column("pct_positive").to_list() == [50.0, 100.0]

    assert result.stats_group is not None
    grouped = result.stats_group.row(0, named=True)
    assert grouped["treatment"] == "dose"
    assert grouped["n_samples"] == 2
    assert math.isclose(grouped["fluor_median_mean"], 17.25)
    assert math.isclose(grouped["fluor_geomean_mean"], (10.0 + math.sqrt(800.0)) / 2.0)
    assert math.isclose(grouped["pct_positive_mean"], 75.0)

    qc = result.qc_table.sort("sample_id")
    assert math.isclose(qc.get_column("pct_nonpositive")[0], 100.0 / 3.0)
    assert qc.get_column("pct_nonpositive")[1] == 0.0
    assert result.threshold_value == 5.0


def test_gate_defaults_use_finite_linear_quantiles() -> None:
    wide = prepare_event_table(
        _tidy_events(),
        channels=("FSC-A", "FSC-H", "SSC-A", "mCherry-A"),
    )

    defaults = gate_defaults(
        wide,
        cells_x_channel="FSC-A",
        cells_y_channel="SSC-A",
        singlet_x_channel="FSC-A",
        singlet_y_channel="FSC-H",
    )

    assert defaults.cells_x.minimum == 10.0
    assert defaults.cells_x.maximum == 1000.0
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(defaults.cells_x.selected, (10.1, 951.5), strict=True)
    )
    assert all(
        math.isclose(actual, expected) for actual, expected in zip(defaults.cells_y.selected, (20.0, 29.9), strict=True)
    )
    assert all(math.isfinite(value) for value in defaults.singlet_ratio.selected)


def test_plot_payload_is_projected_filtered_and_bounded() -> None:
    wide = prepare_event_table(
        _tidy_events(),
        channels=("FSC-A", "FSC-H", "SSC-A", "mCherry-A"),
    )

    payload = prepare_plot_events(
        wide,
        columns=("sample_id", "FSC-A", "SSC-A", "mCherry-A"),
        x_channel="FSC-A",
        y_channel="SSC-A",
        max_events=3,
        group_columns=("sample_id",),
        low_clip_quantile=0.1,
        positive_x=True,
        positive_y=True,
    )

    assert payload.columns == ["sample_id", "FSC-A", "SSC-A", "mCherry-A"]
    assert payload.height <= 3
    assert payload.get_column("FSC-A").is_finite().all()
    assert payload.get_column("SSC-A").is_finite().all()
    assert (payload.get_column("FSC-A") > 0).all()
    assert (payload.get_column("SSC-A") > 0).all()


def test_plot_event_filters_can_use_channels_omitted_from_the_payload() -> None:
    events = pl.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2"],
            "x": [1.0, -1.0, 3.0],
            "y": [2.0, 3.0, float("nan")],
            "signal": [10.0, 20.0, 30.0],
        }
    )

    payload = prepare_plot_events(
        events,
        columns=("sample_id", "signal"),
        x_channel="x",
        y_channel="y",
        max_events=10,
        positive_x=True,
    )

    assert payload.columns == ["sample_id", "signal"]
    assert payload.rows() == [("s1", 10.0)]


def test_plot_event_clipping_can_use_a_group_omitted_from_the_payload() -> None:
    events = pl.DataFrame(
        {
            "batch": ["a", "a", "b", "b"],
            "x": [1.0, 100.0, 2.0, 50.0],
            "y": [1.0, 100.0, 2.0, 50.0],
            "signal": [10.0, 20.0, 30.0, 40.0],
        }
    )

    payload = prepare_plot_events(
        events,
        columns=("x", "y", "signal"),
        x_channel="x",
        y_channel="y",
        max_events=10,
        low_clip_quantile=0.5,
        clip_group_column="batch",
    )

    assert payload.columns == ["x", "y", "signal"]
    assert payload.rows() == [(100.0, 100.0, 20.0), (50.0, 50.0, 40.0)]


def test_plot_event_sampling_can_use_a_group_omitted_from_the_payload() -> None:
    events = pl.DataFrame(
        {
            "sample_id": ["a"] * 9 + ["b"],
            "x": [float(value + 1) for value in range(10)],
            "y": [float(value + 1) for value in range(10)],
            "signal": [float(value) for value in range(9)] + [100.0],
        }
    )

    payload = prepare_plot_events(
        events,
        columns=("x", "y", "signal"),
        x_channel="x",
        y_channel="y",
        max_events=2,
        group_columns=("sample_id",),
    )

    assert payload.columns == ["x", "y", "signal"]
    assert payload.height == 2
    assert 100.0 in payload.get_column("signal")


def test_plot_payload_projection_downsamples_without_changing_source() -> None:
    wide = prepare_event_table(
        _tidy_events(),
        channels=("FSC-A", "FSC-H", "SSC-A", "mCherry-A"),
    )

    payload = prepare_plot_payload(
        wide,
        columns=("sample_id", "FSC-A", "FSC-H"),
        max_events=3,
        group_columns=("sample_id",),
    )

    assert payload.columns == ["sample_id", "FSC-A", "FSC-H"]
    assert payload.height <= 3
    assert wide.height == 6


def test_plot_payload_sampling_can_use_a_group_omitted_from_the_payload() -> None:
    events = pl.DataFrame(
        {
            "sample_id": ["a"] * 9 + ["b"],
            "signal": [float(value) for value in range(9)] + [100.0],
        }
    )

    payload = prepare_plot_payload(
        events,
        columns=("signal",),
        max_events=2,
        group_columns=("sample_id", "missing"),
    )

    assert payload.columns == ["signal"]
    assert payload.height == 2
    assert 100.0 in payload.get_column("signal")


def test_plot_payload_uses_the_existing_seeded_row_selection() -> None:
    frame = pl.DataFrame(
        {
            "event_index": list(range(12)),
            "sample_id": ["a"] * 6 + ["b"] * 6,
            "signal": [float(value) for value in range(12)],
        }
    )
    pandas_frame = frame.to_pandas(use_pyarrow_extension_array=False)
    expected = (
        pandas_frame.groupby(["sample_id"], dropna=False, group_keys=False)
        .apply(lambda group: group.sample(n=3, random_state=0), include_groups=False)
        .reset_index(drop=True)
    )

    actual = prepare_plot_payload(
        frame,
        columns=("event_index", "sample_id", "signal"),
        max_events=6,
        group_columns=("sample_id",),
    )

    assert actual.get_column("event_index").to_list() == expected["event_index"].tolist()
