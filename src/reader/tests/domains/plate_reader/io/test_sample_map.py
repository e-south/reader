from __future__ import annotations

from pathlib import Path

import pandas as pd

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
