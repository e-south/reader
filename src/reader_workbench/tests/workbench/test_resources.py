from __future__ import annotations

from pathlib import Path

from reader_workbench.plugins.ingest.discovery_policy import discover_files


def test_discover_files_excludes_conventional_metadata_names(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "run.xlsx").write_text("x", encoding="utf-8")
    (inputs / "metadata.xlsx").write_text("x", encoding="utf-8")
    (inputs / "sample_map.csv").write_text("x", encoding="utf-8")

    files = discover_files(tmp_path)

    assert files == [(inputs / "run.xlsx").resolve()]


def test_discover_files_can_search_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "raw" / "nested"
    nested.mkdir(parents=True)
    target = nested / "run.xlsx"
    target.write_text("x", encoding="utf-8")

    files = discover_files(tmp_path, recursive=True)

    assert files == [target.resolve()]
