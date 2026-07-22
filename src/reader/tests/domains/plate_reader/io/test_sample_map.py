from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reader.domains.plate_reader.io.sample_map import parse_sample_map


def test_parse_sample_map_normalizes_position_case(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    pd.DataFrame({"Position": ["A1"], "design_id": ["d1"]}).to_csv(path, index=False)

    df = parse_sample_map(str(path))

    assert list(df.columns) == ["position", "design_id"]
    assert df.loc[0, "position"] == "A1"


def test_parse_sample_map_builds_position_from_row_and_col(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    pd.DataFrame({"row": ["B"], "col": [3], "design_id": ["d2"]}).to_csv(path, index=False)

    df = parse_sample_map(str(path))

    assert "row" not in df.columns
    assert "col" not in df.columns
    assert df.loc[0, "position"] == "B3"


@pytest.mark.parametrize("name", ["metadata.xls", "metadata.tsv", "metadata"])
def test_parse_sample_map_rejects_unsupported_formats(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("position,design_id\nA1,d1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected .csv or .xlsx"):
        parse_sample_map(path)


def test_parse_sample_map_rejects_duplicate_normalized_columns(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text("Position,position,design_id\nA1,A1,d1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate column names"):
        parse_sample_map(path)


def test_parse_sample_map_rejects_duplicate_positions(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    pd.DataFrame({"position": ["a1", "A1"], "design_id": ["d1", "d2"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="positions must be unique"):
        parse_sample_map(path)


def test_parse_sample_map_rejects_blank_positions(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    pd.DataFrame({"position": ["A1", None], "design_id": ["d1", "d2"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="blank position values"):
        parse_sample_map(path)
