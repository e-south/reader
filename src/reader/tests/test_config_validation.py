"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_config_validation.py

Validation tests for v2 config loading and read-contract checks.
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
from reader.core.specs import resolve_plot_specs


def _write_config(tmp_path: Path, payload) -> Path:
    path = tmp_path / "config.yaml"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _base_config() -> dict:
    return {
        "schema": "reader/v2",
        "experiment": {"id": "exp_001", "title": "Example"},
        "paths": {"outputs": "./outputs", "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {"specs": []},
        "exports": {"specs": []},
    }


def test_load_rejects_non_mapping(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_load_requires_schema_marker(tmp_path: Path) -> None:
    data = _base_config()
    data.pop("schema")
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="reader/v2"):
        ReaderSpec.load(path)


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    data = _base_config()
    data["schema"] = "reader/v1"
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="reader/v2"):
        ReaderSpec.load(path)


def test_load_rejects_legacy_keys(tmp_path: Path) -> None:
    data = _base_config()
    data["steps"] = []
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="Legacy v1"):
        ReaderSpec.load(path)


def test_load_rejects_plots_steps(tmp_path: Path) -> None:
    data = _base_config()
    data["plots"]["steps"] = []
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="plots.specs"):
        ReaderSpec.load(path)


def test_load_rejects_exports_steps(tmp_path: Path) -> None:
    data = _base_config()
    data["exports"]["steps"] = []
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="exports.specs"):
        ReaderSpec.load(path)


def test_load_requires_pipeline_steps_key(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"].pop("steps")
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_load_rejects_absolute_subdirs(tmp_path: Path) -> None:
    data = _base_config()
    data["paths"]["plots"] = "/tmp/plots"
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_load_expands_file_reads_absolute_paths(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    data_path = tmp_path / "inputs.xlsx"
    data_path.write_text("stub", encoding="utf-8")
    data = _base_config()
    data["pipeline"]["steps"] = [
        {"id": "ingest", "uses": "ingest/synergy_h1", "reads": {"raw": f"file:{data_path}"}}
    ]
    path = _write_config(exp_dir, data)
    spec = ReaderSpec.load(path)
    assert spec.pipeline.steps[0].reads["raw"] == f"file:{data_path.resolve()}"


def test_validate_rejects_unexpected_reads(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [
        {"id": "ingest", "uses": "ingest/synergy_h1"},
        {
            "id": "merge_map",
            "uses": "merge/sample_map",
            "reads": {"df": "ingest/df", "plate_map": "file:./inputs/metadata.xlsx"},
        },
    ]
    path = _write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    with pytest.raises(ConfigError):
        validate_job(spec, console=Console())


def test_validate_rejects_unknown_read_labels(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [
        {"id": "ingest", "uses": "ingest/synergy_h1", "writes": {"df": "raw/df"}},
        {
            "id": "merge_map",
            "uses": "merge/sample_map",
            "reads": {"df": "ingest/df", "sample_map": "file:./inputs/metadata.xlsx"},
        },
    ]
    path = _write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    with pytest.raises(ConfigError):
        validate_job(spec, console=Console())


def test_validate_rejects_duplicate_output_labels(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [
        {"id": "ingest", "uses": "ingest/synergy_h1", "writes": {"df": "shared/df"}},
        {
            "id": "overflow",
            "uses": "transform/overflow_handling",
            "reads": {"df": "shared/df"},
            "writes": {"df": "shared/df"},
        },
    ]
    path = _write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    with pytest.raises(ConfigError):
        validate_job(spec, console=Console())


def test_plot_defaults_apply_to_presets(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [
        {"id": "ingest", "uses": "ingest/synergy_h1", "writes": {"df": "raw/df"}}
    ]
    data["plots"] = {
        "presets": ["plots/plate_reader_yfp_time_series"],
        "defaults": {"reads": {"df": "raw/df"}},
        "specs": [],
    }
    data["exports"] = {"specs": []}
    path = _write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    plot_specs = resolve_plot_specs(spec)
    assert plot_specs
    assert all(ps.reads.get("df") == "raw/df" for ps in plot_specs)
