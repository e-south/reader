from __future__ import annotations

import math

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from reader.domains.cytometry.analysis import (
    CytometryAnalysisError,
    GateSpec,
    ThresholdSpec,
    analyze_events,
    prepare_event_table,
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


def _reference_prepare(frame: pl.DataFrame) -> pd.DataFrame:
    channels = ["FSC-A", "FSC-H", "SSC-A", "mCherry-A"]
    work = frame.filter(pl.col("channel").is_in(channels))
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


def test_prepare_event_table_matches_reference_reduction_and_projects_before_collect() -> None:
    actual = prepare_event_table(
        _tidy_events().lazy(),
        channels=("FSC-A", "FSC-H", "SSC-A", "mCherry-A"),
    )
    expected = _reference_prepare(_tidy_events())
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
    assert actual.get_column("event_index").to_list() == [0, 1, 2, 0, 1, 2]
    assert "unused" not in actual.columns


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


def test_prepare_event_table_rejects_metadata_drift_within_one_event() -> None:
    events = (
        _tidy_events()
        .with_row_index("row_number")
        .with_columns(
            pl.when(pl.col("row_number") == 0)
            .then(pl.lit("drifted-design"))
            .otherwise(pl.col("design_id"))
            .alias("design_id")
        )
    )

    with pytest.raises(
        CytometryAnalysisError,
        match=r"inconsistent metadata.*sample_id.*event_index.*design_id",
    ):
        prepare_event_table(
            events.drop("row_number"),
            channels=("FSC-A", "FSC-H", "SSC-A", "mCherry-A"),
        )


def test_prepare_event_table_rejects_missing_selected_channel_within_one_event() -> None:
    events = _tidy_events().filter(
        ~(pl.col("sample_id").eq("s1") & pl.col("event_index").eq(0) & pl.col("channel").eq("mCherry-A"))
    )

    with pytest.raises(
        CytometryAnalysisError,
        match=r"missing selected channels.*sample_id.*event_index.*mCherry-A",
    ):
        prepare_event_table(
            events.lazy(),
            channels=("FSC-A", "FSC-H", "SSC-A", "mCherry-A"),
        )


def test_gate_and_summary_values_match_declared_analysis_semantics() -> None:
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
        group_column="treatment",
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
