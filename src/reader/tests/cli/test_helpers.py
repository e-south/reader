from __future__ import annotations

from pathlib import Path

import pytest

from reader.errors import ReaderError, RecordError
from reader.protocols.model import binding_value
from reader.runtime import builtin_runtime
from reader.workbench.cli.helpers import append_journal, dataframe_record_contracts
from reader.workbench.cli.shared import json_friendly


def test_dataframe_record_contracts_raises_on_corrupt_catalog(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    manifests = outputs / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "records.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(RecordError, match="not valid JSON"):
        dataframe_record_contracts(outputs, runtime=builtin_runtime())


def test_json_friendly_serializes_protocol_binding_value_ref() -> None:
    assert json_friendly(binding_value("sample_map")) == {"binding_value": "sample_map"}
    assert json_friendly(binding_value("sample_map", default="metadata.xlsx")) == {
        "binding_value": "sample_map",
        "default": "metadata.xlsx",
    }


def test_append_journal_migrates_legacy_lowercase_filename(tmp_path: Path) -> None:
    job_path = tmp_path / "config.yaml"
    job_path.write_text("schema: reader/v7\n", encoding="utf-8")
    legacy = tmp_path / "journal.md"
    legacy.write_text("# Experiment Journal\n\nlegacy entry\n", encoding="utf-8")

    append_journal(job_path, "uv run reader run config.yaml")

    canonical = tmp_path / "JOURNAL.md"
    assert canonical.exists()
    text = canonical.read_text(encoding="utf-8")
    assert "legacy entry" in text
    assert "uv run reader run config.yaml" in text


def test_append_journal_rejects_split_case_journal_files(monkeypatch, tmp_path: Path) -> None:
    job_path = tmp_path / "config.yaml"
    job_path.write_text("schema: reader/v7\n", encoding="utf-8")
    (tmp_path / "JOURNAL.md").write_text("# Experiment Journal\n", encoding="utf-8")
    (tmp_path / "journal.md").write_text("# Experiment Journal\n", encoding="utf-8")
    monkeypatch.setattr(Path, "samefile", lambda self, other: False)

    with pytest.raises(ReaderError, match="Both JOURNAL.md and journal.md exist"):
        append_journal(job_path, "uv run reader run config.yaml")
