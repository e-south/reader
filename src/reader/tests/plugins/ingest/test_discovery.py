from __future__ import annotations

import os
from pathlib import Path

import pytest

from reader.core.errors import ParseError
from reader.plugins.ingest._discovery import discover_auto_input_files


def _touch(path: Path, *, mtime: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_discover_auto_input_files_picks_latest(tmp_path: Path):
    older = _touch(tmp_path / "inputs" / "old.xlsx", mtime=1)
    newer = _touch(tmp_path / "inputs" / "new.xlsx", mtime=2)

    files = discover_auto_input_files(
        exp_dir=tmp_path,
        auto_roots=None,
        auto_include=["*.xlsx"],
        auto_exclude=[],
        auto_recursive=False,
        auto_pick="latest",
        discovery_label="raw .xlsx files",
        singular_label="workbook",
    )

    assert files == [newer]
    assert older.exists()


def test_discover_auto_input_files_single_mode_requires_exactly_one(tmp_path: Path):
    _touch(tmp_path / "inputs" / "a.xlsx")
    _touch(tmp_path / "inputs" / "b.xlsx")

    with pytest.raises(ParseError, match="exactly one workbook"):
        discover_auto_input_files(
            exp_dir=tmp_path,
            auto_roots=None,
            auto_include=["*.xlsx"],
            auto_exclude=[],
            auto_recursive=False,
            auto_pick="single",
            discovery_label="raw .xlsx files",
            singular_label="workbook",
        )
