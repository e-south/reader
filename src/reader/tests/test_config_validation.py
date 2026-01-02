"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_config_validation.py

Validation tests for config loading and read-contract checks.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from rich.console import Console

from reader.core.config_model import ReaderSpec
from reader.core.engine import validate as validate_job
from reader.core.errors import ConfigError


def _write_config(tmp_path: Path, payload) -> Path:
    path = tmp_path / "config.yaml"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_rejects_non_mapping(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_load_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    data = {
        "experiment": {"outputs": "./outputs"},
        "steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}],
        "bogus": 1,
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_overrides_require_known_step_ids(tmp_path: Path) -> None:
    data = {
        "experiment": {"outputs": "./outputs"},
        "steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}],
        "overrides": {"missing": {"with": {"mode": "auto"}}},
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_overrides_cannot_change_step_id(tmp_path: Path) -> None:
    data = {
        "experiment": {"outputs": "./outputs"},
        "steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}],
        "overrides": {"ingest": {"id": "renamed"}},
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_load_accepts_overrides_and_deliverable_presets(tmp_path: Path) -> None:
    data = {
        "experiment": {"outputs": "./outputs"},
        "steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}],
        "overrides": {"ingest": {"with": {"mode": "auto"}}},
        "deliverable_presets": ["plots/plate_reader_yfp_snapshots"],
    }
    path = _write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    assert spec.steps[0].with_["mode"] == "auto"


def test_validate_rejects_unexpected_reads(tmp_path: Path) -> None:
    data = {
        "experiment": {"outputs": "./outputs"},
        "steps": [
            {"id": "ingest", "uses": "ingest/synergy_h1"},
            {
                "id": "merge_map",
                "uses": "merge/sample_map",
                "reads": {"df": "ingest/df", "plate_map": "file:./metadata.xlsx"},
            },
        ],
    }
    path = _write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    with pytest.raises(ConfigError):
        validate_job(spec, console=Console())


def test_validate_rejects_unknown_read_labels(tmp_path: Path) -> None:
    data = {
        "experiment": {"outputs": "./outputs"},
        "steps": [
            {"id": "ingest", "uses": "ingest/synergy_h1", "writes": {"df": "raw/df"}},
            {
                "id": "merge_map",
                "uses": "merge/sample_map",
                "reads": {"df": "ingest/df", "sample_map": "file:./metadata.xlsx"},
            },
        ],
    }
    path = _write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    with pytest.raises(ConfigError):
        validate_job(spec, console=Console())


def test_validate_rejects_duplicate_output_labels(tmp_path: Path) -> None:
    data = {
        "experiment": {"outputs": "./outputs"},
        "steps": [
            {"id": "ingest", "uses": "ingest/synergy_h1", "writes": {"df": "shared/df"}},
            {
                "id": "overflow",
                "uses": "transform/overflow_handling",
                "reads": {"df": "shared/df"},
                "writes": {"df": "shared/df"},
            },
        ],
    }
    path = _write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    with pytest.raises(ConfigError):
        validate_job(spec, console=Console())
