from __future__ import annotations

import struct
from pathlib import Path

import pandas as pd
import yaml

from reader.core.artifacts import ArtifactStore
from reader.core.engine import run_job


def _write_cfg(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _write_synergy_xlsx(path: Path) -> None:
    rows = [
        [None, "2025-01-01", "Date"],
        [None, "00:00:00", "Time"],
        [None, None, None],
        ["OD600", None, None],
        [None, "Time", "A1"],
        [None, "0", 0.10],
        [None, "1", 0.20],
    ]
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False, header=False)


def _write_minimal_fcs(path: Path) -> None:
    par = 2
    tot = 3
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data_bytes = struct.pack("<" + "f" * len(data), *data)

    text_items = {
        "$BEGINANALYSIS": "0",
        "$ENDANALYSIS": "0",
        "$BEGINSTEXT": "0",
        "$ENDSTEXT": "0",
        "$BEGINDATA": "0",
        "$ENDDATA": "0",
        "$BYTEORD": "1,2,3,4",
        "$DATATYPE": "F",
        "$MODE": "L",
        "$NEXTDATA": "0",
        "$PAR": str(par),
        "$TOT": str(tot),
        "$P1B": "32",
        "$P1N": "FSC-A",
        "$P1R": "1024",
        "$P2B": "32",
        "$P2N": "SSC-A",
        "$P2R": "1024",
    }
    keys = list(text_items.keys())
    delim = "|"

    def build_text(items: dict[str, str]) -> bytes:
        text = delim + delim.join([f"{k}{delim}{items[k]}" for k in keys]) + delim
        return text.encode("ascii")

    header_len = 58
    text_bytes = build_text(text_items)
    text_start = header_len
    text_end = text_start + len(text_bytes) - 1
    data_start = text_end + 1
    data_end = data_start + len(data_bytes) - 1

    text_items["$BEGINSTEXT"] = str(text_start)
    text_items["$ENDSTEXT"] = str(text_end)
    text_items["$BEGINDATA"] = str(data_start)
    text_items["$ENDDATA"] = str(data_end)
    text_bytes = build_text(text_items)
    text_end = text_start + len(text_bytes) - 1
    data_start = text_end + 1
    data_end = data_start + len(data_bytes) - 1

    text_items["$BEGINSTEXT"] = str(text_start)
    text_items["$ENDSTEXT"] = str(text_end)
    text_items["$BEGINDATA"] = str(data_start)
    text_items["$ENDDATA"] = str(data_end)
    text_bytes = build_text(text_items)
    text_end = text_start + len(text_bytes) - 1
    data_start = text_end + 1
    data_end = data_start + len(data_bytes) - 1

    def fmt(n: int) -> str:
        return str(n).rjust(8)

    header = (
        "FCS3.0"
        + "    "
        + fmt(text_start)
        + fmt(text_end)
        + fmt(data_start)
        + fmt(data_end)
        + fmt(0)
        + fmt(0)
    )
    header_bytes = header.encode("ascii")
    if len(header_bytes) != 58:
        raise AssertionError("Invalid FCS header length")

    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(text_bytes)
        f.write(data_bytes)


def test_pipeline_synergy_h1_minimal(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    plate = inputs / "plate.xlsx"
    _write_synergy_xlsx(plate)

    meta = tmp_path / "metadata.csv"
    meta.write_text("position,design_id,treatment\nA1,designA,cond\n", encoding="utf-8")

    cfg = {
        "experiment": {"outputs": "./outputs"},
        "steps": [
            {
                "id": "ingest",
                "uses": "ingest/synergy_h1",
                "reads": {"raw": "file:./inputs/plate.xlsx"},
                "with": {
                    "mode": "kinetic_only",
                    "channels": ["OD600"],
                    "sheet_names": ["Sheet1"],
                    "add_sheet": False,
                    "print_summary": False,
                },
            },
            {
                "id": "merge_map",
                "uses": "merge/sample_map",
                "reads": {"df": "ingest/df", "sample_map": "file:./metadata.csv"},
                "with": {"require_columns": ["design_id", "treatment"], "require_non_null": True},
            },
        ],
    }
    cfg_path = _write_cfg(tmp_path / "config.yaml", cfg)

    run_job(cfg_path, log_level="ERROR")

    store = ArtifactStore(tmp_path / "outputs")
    df = store.read("merge_map/df").load_dataframe()
    assert {"position", "design_id", "treatment"}.issubset(df.columns)


def test_pipeline_flow_cytometer_minimal(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    fcs_path = inputs / "sample.fcs"
    _write_minimal_fcs(fcs_path)

    meta = tmp_path / "metadata.csv"
    meta.write_text("sample_id,design_id,treatment\nsample,designA,cond\n", encoding="utf-8")

    cfg = {
        "experiment": {"outputs": "./outputs"},
        "steps": [
            {
                "id": "ingest",
                "uses": "ingest/flow_cytometer",
                "reads": {"raw": "file:./inputs/sample.fcs"},
                "with": {"channel_name_field": "pnn", "print_summary": False},
            },
            {
                "id": "merge_metadata",
                "uses": "merge/sample_metadata",
                "reads": {"df": "ingest/df", "metadata": "file:./metadata.csv"},
                "with": {"require_columns": ["design_id", "treatment"], "require_non_null": True},
            },
        ],
    }
    cfg_path = _write_cfg(tmp_path / "config.yaml", cfg)

    run_job(cfg_path, log_level="ERROR")

    store = ArtifactStore(tmp_path / "outputs")
    df = store.read("merge_metadata/df").load_dataframe()
    assert {"sample_id", "design_id", "treatment"}.issubset(df.columns)
