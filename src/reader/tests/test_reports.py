from __future__ import annotations

import io
from pathlib import Path

import pytest
import yaml
from rich.console import Console

from reader.core.config_model import ReaderSpec
from reader.core.engine import validate
from reader.core.errors import ConfigError


def _write_cfg(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _validate_spec(spec: ReaderSpec) -> None:
    validate(spec, console=Console(file=io.StringIO(), force_terminal=False, width=120))


def test_reports_allow_export_from_pipeline(tmp_path: Path) -> None:
    payload = {
        "experiment": {"outputs": "./outputs"},
        "steps": [
            {"id": "ingest", "uses": "ingest/synergy_h1", "reads": {"raw": "file:./dummy.xlsx"}},
        ],
        "reports": [
            {
                "id": "export_csv",
                "uses": "export/csv",
                "reads": {"df": "ingest/df"},
                "with": {"path": "exports/out.csv"},
            }
        ],
    }
    spec = ReaderSpec.load(_write_cfg(tmp_path, payload))
    _validate_spec(spec)


def test_reports_reject_file_reads(tmp_path: Path) -> None:
    payload = {
        "experiment": {"outputs": "./outputs"},
        "steps": [
            {"id": "ingest", "uses": "ingest/synergy_h1"},
        ],
        "reports": [
            {
                "id": "bad_export",
                "uses": "export/csv",
                "reads": {"df": "file:./data.csv"},
                "with": {"path": "exports/out.csv"},
            }
        ],
    }
    spec = ReaderSpec.load(_write_cfg(tmp_path, payload))
    with pytest.raises(ConfigError):
        _validate_spec(spec)


def test_reports_reject_non_report_plugins(tmp_path: Path) -> None:
    payload = {
        "experiment": {"outputs": "./outputs"},
        "steps": [
            {"id": "ingest", "uses": "ingest/synergy_h1"},
        ],
        "reports": [
            {
                "id": "bad_transform",
                "uses": "transform/ratio",
                "reads": {"df": "ingest/df"},
                "with": {"name": "A/B", "numerator": "A", "denominator": "B"},
            }
        ],
    }
    spec = ReaderSpec.load(_write_cfg(tmp_path, payload))
    with pytest.raises(ConfigError):
        _validate_spec(spec)
