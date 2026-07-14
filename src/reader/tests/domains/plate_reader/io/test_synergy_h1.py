from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reader.domains.plate_reader.io.synergy_h1 import (
    parse_kinetic_only,
    parse_snapshot_and_timeseries,
)


def _write_workbook(path: Path, rows: list[list[object]], *, sheet_name: str = "Plate 1") -> Path:
    pd.DataFrame(rows).to_excel(path, sheet_name=sheet_name, header=False, index=False)
    return path


def _snapshot_rows(
    *,
    first_label: object = "OD600:600",
    second_label: object = "YFP:500,530",
    first_value: object = "1.5",
    second_value: object = "20",
) -> list[list[object]]:
    return [
        ["Date", "2026-07-07"],
        ["Time", "12:00:00"],
        ["Results"],
        [],
        [None, None, "1", "2"],
        [None, "A", first_value, "2.5", first_label],
        [None, None, second_value, "21", second_label],
    ]


def _kinetic_rows(*, value: object) -> list[list[object]]:
    return [
        ["Date", "2026-07-07"],
        ["Time", "12:00:00"],
        ["OD600:600"],
        [],
        [None, "Time", "A1"],
        [None, "00:00:00", value],
    ]


def test_snapshot_channel_identity_comes_from_workbook_labels(tmp_path: Path) -> None:
    workbook = _write_workbook(tmp_path / "snapshot.xlsx", _snapshot_rows())

    result = parse_snapshot_and_timeseries(
        workbook,
        channels=["YFP", "OD600"],
        channel_map={"OD600:600": "OD600", "YFP:500,530": "YFP"},
        include_kinetic=False,
    )

    a1 = result.loc[result["position"] == "A1", ["channel", "value"]].set_index("channel")["value"]
    assert a1.to_dict() == {"OD600": 1.5, "YFP": 20.0}
    assert {"sheet_index", "sheet_name"} <= set(result.columns)
    assert "sheet" not in result.columns


def test_snapshot_label_only_row_is_not_misclassified_as_kinetic(tmp_path: Path) -> None:
    rows = _snapshot_rows()
    rows[-1][2:4] = [None, None]
    workbook = _write_workbook(tmp_path / "snapshot.xlsx", rows)

    result = parse_snapshot_and_timeseries(
        workbook,
        channels=["OD600"],
        channel_map={"OD600:600": "OD600", "YFP:500,530": "YFP"},
        include_kinetic=False,
    )

    assert set(result["channel"]) == {"OD600"}


def test_snapshot_requires_an_explicit_channel_map(tmp_path: Path) -> None:
    workbook = _write_workbook(tmp_path / "snapshot.xlsx", _snapshot_rows())

    with pytest.raises(ValueError, match="Snapshot parsing requires an explicit 'channel_map'"):
        parse_snapshot_and_timeseries(
            workbook,
            channels=["OD600", "YFP"],
            include_kinetic=False,
        )


def test_snapshot_rejects_missing_channel_label(tmp_path: Path) -> None:
    workbook = _write_workbook(
        tmp_path / "snapshot.xlsx",
        _snapshot_rows(second_label=None),
        sheet_name="Assay",
    )

    with pytest.raises(ValueError, match=r"Missing snapshot channel label.*sheet 'Assay'.*plate row 'A'"):
        parse_snapshot_and_timeseries(
            workbook,
            channels=["OD600", "YFP"],
            channel_map={"OD600:600": "OD600", "YFP:500,530": "YFP"},
            include_kinetic=False,
        )


def test_snapshot_rejects_duplicate_channel_label_within_plate_row(tmp_path: Path) -> None:
    workbook = _write_workbook(
        tmp_path / "snapshot.xlsx",
        _snapshot_rows(second_label="OD600:600"),
        sheet_name="Assay",
    )

    with pytest.raises(ValueError, match=r"Duplicate snapshot channel 'OD600'.*sheet 'Assay'.*plate row 'A'"):
        parse_snapshot_and_timeseries(
            workbook,
            channels=["OD600"],
            channel_map={"OD600:600": "OD600"},
            include_kinetic=False,
        )


def test_snapshot_rejects_nonexact_channel_label(tmp_path: Path) -> None:
    workbook = _write_workbook(
        tmp_path / "snapshot.xlsx",
        _snapshot_rows(first_label="YFP measurement"),
        sheet_name="Assay",
    )

    with pytest.raises(ValueError, match=r"Missing snapshot channel label.*sheet 'Assay'.*plate row 'A'"):
        parse_snapshot_and_timeseries(
            workbook,
            channels=["yellow"],
            channel_map={"FP": "generic", "YFP": "yellow"},
            include_kinetic=False,
        )


def test_snapshot_rejects_nonnumeric_measurement_with_context(tmp_path: Path) -> None:
    workbook = _write_workbook(
        tmp_path / "snapshot.xlsx",
        _snapshot_rows(first_value="not-a-reading"),
        sheet_name="Assay",
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid snapshot measurement token 'not-a-reading'.*sheet 'Assay'.*channel 'OD600'.*well 'A1'",
    ):
        parse_snapshot_and_timeseries(
            workbook,
            channels=["OD600", "YFP"],
            channel_map={"OD600:600": "OD600", "YFP:500,530": "YFP"},
            include_kinetic=False,
        )


def test_snapshot_preserves_declared_overflow_tokens(tmp_path: Path) -> None:
    workbook = _write_workbook(
        tmp_path / "snapshot.xlsx",
        _snapshot_rows(first_value="OVERFLOW"),
    )

    result = parse_snapshot_and_timeseries(
        workbook,
        channels=["OD600", "YFP"],
        channel_map={"OD600:600": "OD600", "YFP:500,530": "YFP"},
        include_kinetic=False,
    )

    reading = result.loc[(result["position"] == "A1") & (result["channel"] == "OD600")].iloc[0]
    assert reading["value"] == float("inf")
    assert bool(reading["overflow"])


def test_kinetic_rejects_nonnumeric_measurement_with_context(tmp_path: Path) -> None:
    workbook = _write_workbook(
        tmp_path / "kinetic.xlsx",
        _kinetic_rows(value="not-a-reading"),
        sheet_name="Assay",
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid kinetic measurement token 'not-a-reading'.*sheet 'Assay'.*channel 'OD600'.*well 'A1'",
    ):
        parse_kinetic_only(
            workbook,
            channels=["OD600"],
            channel_map={"OD600:600": "OD600"},
        )


def test_kinetic_rejects_nonexact_channel_label(tmp_path: Path) -> None:
    rows = _kinetic_rows(value="1.0")
    rows[2] = ["YFP measurement"]
    workbook = _write_workbook(tmp_path / "kinetic.xlsx", rows, sheet_name="Assay")

    with pytest.raises(ValueError, match=r"No kinetic data found in sheet 'Assay'"):
        parse_kinetic_only(
            workbook,
            channels=["yellow"],
            channel_map={"FP": "generic", "YFP": "yellow"},
        )


def test_kinetic_rejects_one_invalid_time_without_dropping_its_measurement(tmp_path: Path) -> None:
    rows = _kinetic_rows(value="1.0")
    rows.extend([[None, "bad-time", "2.0"], [None, "00:10:00", "3.0"]])
    workbook = _write_workbook(tmp_path / "kinetic.xlsx", rows, sheet_name="Assay")

    with pytest.raises(
        ValueError,
        match=r"Invalid kinetic time token 'bad-time'.*sheet 'Assay'.*channel 'OD600'",
    ):
        parse_kinetic_only(
            workbook,
            channels=["OD600"],
            channel_map={"OD600:600": "OD600"},
        )


@pytest.mark.parametrize("token", ["leftover", "infinite", "uncovered"])
def test_nonnumeric_substrings_are_not_treated_as_overflow(tmp_path: Path, token: str) -> None:
    workbook = _write_workbook(
        tmp_path / "kinetic.xlsx",
        _kinetic_rows(value=token),
        sheet_name="Assay",
    )

    with pytest.raises(ValueError, match=rf"Invalid kinetic measurement token '{token}'"):
        parse_kinetic_only(
            workbook,
            channels=["OD600"],
            channel_map={"OD600:600": "OD600"},
        )


def test_kinetic_rejects_duplicate_measurement_keys(tmp_path: Path) -> None:
    rows = _kinetic_rows(value="1.0")
    rows.append([None, "00:00:00", "2.0"])
    workbook = _write_workbook(tmp_path / "kinetic.xlsx", rows, sheet_name="Assay")

    with pytest.raises(ValueError, match="must be unique"):
        parse_kinetic_only(
            workbook,
            channels=["OD600"],
            channel_map={"OD600:600": "OD600"},
        )


@pytest.mark.parametrize(
    ("well", "message"), [("A13", "outside the 96-well range"), ("A0", "outside the 96-well range")]
)
def test_kinetic_rejects_out_of_range_well_headers(tmp_path: Path, well: str, message: str) -> None:
    rows = _kinetic_rows(value="1.0")
    rows[4] = [None, "Time", well]
    workbook = _write_workbook(tmp_path / "kinetic.xlsx", rows, sheet_name="Assay")

    with pytest.raises(ValueError, match=message):
        parse_kinetic_only(workbook, channel_map={"OD600:600": "OD600"})


def test_kinetic_rejects_duplicate_well_headers(tmp_path: Path) -> None:
    rows = _kinetic_rows(value="1.0")
    rows[4] = [None, "Time", "A1", "A1"]
    rows[5] = [None, "00:00:00", "1.0", "2.0"]
    workbook = _write_workbook(tmp_path / "kinetic.xlsx", rows, sheet_name="Assay")

    with pytest.raises(ValueError, match="duplicate well columns"):
        parse_kinetic_only(workbook, channel_map={"OD600:600": "OD600"})


@pytest.mark.parametrize(("well", "message"), [("13", "outside the 96-well range"), ("1", "duplicate well columns")])
def test_snapshot_rejects_invalid_well_headers(tmp_path: Path, well: str, message: str) -> None:
    rows = _snapshot_rows()
    rows[4][3] = well
    workbook = _write_workbook(tmp_path / "snapshot.xlsx", rows, sheet_name="Assay")

    with pytest.raises(ValueError, match=message):
        parse_snapshot_and_timeseries(
            workbook,
            channel_map={"OD600:600": "OD600", "YFP:500,530": "YFP"},
            include_kinetic=False,
        )


def test_kinetic_normalizes_biotek_a_b_channel_suffix(tmp_path: Path) -> None:
    rows = _kinetic_rows(value="1.0")
    rows[2] = ["YFP B:500,530"]
    workbook = _write_workbook(tmp_path / "kinetic.xlsx", rows, sheet_name="Assay")

    result = parse_kinetic_only(
        workbook,
        channel_map={"YFP:500,530": "YFP"},
    )

    assert set(result["channel"]) == {"YFP"}


def test_kinetic_rejects_wrong_wavelength_declaration(tmp_path: Path) -> None:
    rows = _kinetic_rows(value="1.0")
    rows[2] = ["YFP B:999,999"]
    workbook = _write_workbook(tmp_path / "kinetic.xlsx", rows, sheet_name="Assay")

    with pytest.raises(ValueError, match="No kinetic data found"):
        parse_kinetic_only(
            workbook,
            channel_map={"YFP:500,530": "YFP"},
        )


def test_kinetic_requires_all_explicitly_mapped_channels_even_when_names_contain_slashes(tmp_path: Path) -> None:
    workbook = _write_workbook(tmp_path / "kinetic.xlsx", _kinetic_rows(value="1.0"), sheet_name="Assay")

    with pytest.raises(ValueError, match=r"Kinetic data missing for channels: \['YFP/CFP'\]"):
        parse_kinetic_only(
            workbook,
            channel_map={"OD600:600": "OD600", "YFP:500,530": "YFP/CFP"},
        )


def test_kinetic_stops_at_following_results_section(tmp_path: Path) -> None:
    rows = _kinetic_rows(value="1.0")
    rows.extend(
        [
            ["Results"],
            [],
            [None, None, "1"],
            [None, "A", "2.0", "OD600:600"],
        ]
    )
    workbook = _write_workbook(tmp_path / "kinetic.xlsx", rows, sheet_name="Assay")

    result = parse_kinetic_only(
        workbook,
        channel_map={"OD600:600": "OD600"},
    )

    assert result[["position", "channel", "value"]].to_dict(orient="records") == [
        {"position": "A1", "channel": "OD600", "value": 1.0}
    ]


def test_mixed_parse_requires_each_channel_in_each_requested_source(tmp_path: Path) -> None:
    rows = _snapshot_rows()
    rows.extend(
        [
            ["OD600 B:600"],
            [],
            [None, "Time", "A1"],
            [None, "00:00:00", "1.0"],
            [None, "00:10:00", "1.1"],
        ]
    )
    workbook = _write_workbook(tmp_path / "mixed.xlsx", rows, sheet_name="Assay")

    with pytest.raises(ValueError, match=r"Kinetic data missing for channels: \['YFP'\]"):
        parse_snapshot_and_timeseries(
            workbook,
            channel_map={"OD600:600": "OD600", "YFP:500,530": "YFP"},
            include_snapshot=True,
            include_kinetic=True,
        )


def test_mixed_parse_rejects_kinetic_source_removed_by_initial_overlap(tmp_path: Path) -> None:
    rows = _snapshot_rows()[:-1]
    rows.extend(
        [
            ["OD600 B:600"],
            [],
            [None, "Time", "A1"],
            [None, "00:00:00", "1.0"],
        ]
    )
    workbook = _write_workbook(tmp_path / "mixed.xlsx", rows, sheet_name="Assay")

    with pytest.raises(ValueError, match=r"Missing requested Synergy data sources: \['kinetic'\]"):
        parse_snapshot_and_timeseries(
            workbook,
            channel_map={"OD600:600": "OD600"},
            include_snapshot=True,
            include_kinetic=True,
        )


def test_mixed_parse_keeps_later_kinetic_measurements_after_initial_overlap(tmp_path: Path) -> None:
    rows = _snapshot_rows()[:-1]
    rows.extend(
        [
            ["OD600 B:600"],
            [],
            [None, "Time", "A1"],
            [None, "00:00:00", "1.0"],
            [None, "00:10:00", "1.1"],
        ]
    )
    workbook = _write_workbook(tmp_path / "mixed.xlsx", rows, sheet_name="Assay")

    result = parse_snapshot_and_timeseries(
        workbook,
        channel_map={"OD600:600": "OD600"},
        include_snapshot=True,
        include_kinetic=True,
    )

    assert set(result["source"]) == {"snapshot", "kinetic"}
    assert result.loc[result["source"] == "kinetic", "time"].tolist() == pytest.approx([1 / 6])


@pytest.mark.parametrize("suffix", [".xls", ".csv"])
def test_synergy_parser_accepts_only_xlsx(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"raw{suffix}"
    path.touch()

    with pytest.raises(ValueError, match=r"requires a modern \.xlsx workbook"):
        parse_kinetic_only(path, channel_map={"OD600:600": "OD600"})


def test_synergy_parser_requires_a_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "raw.xlsx"
    path.mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        parse_kinetic_only(path, channel_map={"OD600:600": "OD600"})
