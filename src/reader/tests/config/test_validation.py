"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/config/test_validation.py

Validation tests for v3 config loading and read-contract checks.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from reader.core.config import ReaderSpec
from reader.core.engine import validate as validate_job
from reader.core.errors import ConfigError
from reader.core.workbench import resolve_workbench
from reader.tests.support import base_reader_config, write_config


def _base_config() -> dict:
    return base_reader_config(experiment_id="exp_001", title="Example")


def test_load_rejects_non_mapping(tmp_path: Path) -> None:
    path = write_config(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_load_requires_schema_marker(tmp_path: Path) -> None:
    data = _base_config()
    data.pop("schema")
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="reader/v3"):
        ReaderSpec.load(path)


def test_load_derives_experiment_id_from_directory_name_when_missing(tmp_path: Path) -> None:
    exp_dir = tmp_path / "20269999_example_exp"
    exp_dir.mkdir()
    data = _base_config()
    data["experiment"] = {}
    path = write_config(exp_dir, data)

    spec = ReaderSpec.load(path)

    assert spec.experiment.id == "20269999_example_exp"
    assert spec.experiment.title == "20269999_example_exp"


def test_load_derives_experiment_fields_when_experiment_block_is_omitted(tmp_path: Path) -> None:
    exp_dir = tmp_path / "20269999_example_exp"
    exp_dir.mkdir()
    data = _base_config()
    data.pop("experiment")
    path = write_config(exp_dir, data)

    spec = ReaderSpec.load(path)

    assert spec.experiment.id == "20269999_example_exp"
    assert spec.experiment.title == "20269999_example_exp"


def test_load_derives_title_from_id_when_missing(tmp_path: Path) -> None:
    data = _base_config()
    data["experiment"] = {"id": "exp_alpha"}
    path = write_config(tmp_path, data)

    spec = ReaderSpec.load(path)

    assert spec.experiment.id == "exp_alpha"
    assert spec.experiment.title == "exp_alpha"


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    data = _base_config()
    data["schema"] = "reader/v1"
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="reader/v3"):
        ReaderSpec.load(path)


def test_load_rejects_legacy_keys(tmp_path: Path) -> None:
    data = _base_config()
    data["steps"] = []
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="Unsupported legacy/removed"):
        ReaderSpec.load(path)


def test_load_rejects_legacy_notebook_key(tmp_path: Path) -> None:
    data = _base_config()
    data["notebook"] = {"preset": "notebook/eda"}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="Unsupported legacy/removed"):
        ReaderSpec.load(path)


def test_load_rejects_removed_data_section(tmp_path: Path) -> None:
    data = _base_config()
    data["data"] = {"aliases": {}}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="Unsupported legacy/removed"):
        ReaderSpec.load(path)


def test_load_rejects_semantics_groupings_key(tmp_path: Path) -> None:
    data = _base_config()
    data["semantics"] = {"groupings": {}}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="semantics.groups"):
        ReaderSpec.load(path)


def test_load_rejects_legacy_semantics_orderings(tmp_path: Path) -> None:
    data = _base_config()
    data["semantics"] = {"orderings": {"treatment": "bad"}}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="assay.orders"):
        ReaderSpec.load(path)


def test_load_normalizes_assay_orders_to_string_lists(tmp_path: Path) -> None:
    data = _base_config()
    data["assay"] = {"orders": {"states": {"column": "treatment_alias", "values": ["A", 2, True]}}}
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    assert spec.assay.orders["states"].column == "treatment_alias"
    assert spec.assay.orders["states"].values == ["A", "2", "True"]


def test_load_normalizes_resource_paths(tmp_path: Path) -> None:
    data = _base_config()
    data["resources"] = {"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}}
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    assert spec.resources.by_id["sample_map"].kind == "file"
    assert spec.resources.by_id["sample_map"].path == str((tmp_path / "inputs/metadata.xlsx").resolve())


def test_load_rejects_plots_steps(tmp_path: Path) -> None:
    data = _base_config()
    data["plots"]["steps"] = []
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="plots.specs"):
        ReaderSpec.load(path)


def test_load_rejects_exports_steps(tmp_path: Path) -> None:
    data = _base_config()
    data["exports"]["steps"] = []
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="exports.specs"):
        ReaderSpec.load(path)


def test_load_requires_pipeline_steps_key(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"].pop("steps")
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_load_rejects_absolute_subdirs(tmp_path: Path) -> None:
    data = _base_config()
    data["paths"]["plots"] = "/tmp/plots"
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_workbench_resolves_file_reads_absolute_paths(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    data_path = tmp_path / "inputs.xlsx"
    data_path.write_text("stub", encoding="utf-8")
    data = _base_config()
    data["pipeline"]["steps"] = [{"id": "ingest", "uses": "ingest/synergy_h1", "reads": {"raw": f"file:{data_path}"}}]
    path = write_config(exp_dir, data)
    spec = ReaderSpec.load(path)
    assert resolve_workbench(spec).pipeline[0].reads["raw"] == f"file:{data_path.resolve()}"


def test_workbench_resolves_resource_reads_to_files(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    data_path = exp_dir / "inputs.xlsx"
    data_path.write_text("stub", encoding="utf-8")
    data = _base_config()
    data["resources"] = {"sample_map": {"kind": "file", "path": "./inputs.xlsx"}}
    data["pipeline"]["steps"] = [
        {"id": "merge", "uses": "merge/sample_map", "reads": {"df": "raw/df", "sample_map": "resource:sample_map"}}
    ]
    path = write_config(exp_dir, data)
    spec = ReaderSpec.load(path)
    assert resolve_workbench(spec).pipeline[0].reads["sample_map"] == f"file:{data_path.resolve()}"


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
    path = write_config(tmp_path, data)
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
            "reads": {"df": "ingest/df", "sample_map": "resource:sample_map"},
        },
    ]
    data["resources"] = {"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}}
    path = write_config(tmp_path, data)
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
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    with pytest.raises(ConfigError):
        validate_job(spec, console=Console())


def test_plot_defaults_apply_to_presets(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [{"id": "ingest", "uses": "ingest/synergy_h1", "writes": {"df": "raw/df"}}]
    data["plots"] = {
        "presets": ["plots/plate_reader_yfp_time_series"],
        "defaults": {"reads": {"df": "raw/df"}},
        "specs": [],
    }
    data["exports"] = {"specs": []}
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    plot_specs = resolve_workbench(spec).plots
    assert plot_specs
    assert all(ps.reads.get("df") == "raw/df" for ps in plot_specs)


def test_pipeline_presets_expand_dual_reporter_screen_base(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"] = {
        "presets": ["plate_reader/synergy_h1", "plate_reader/dual_reporter_screen_base"],
        "steps": [],
    }
    data["resources"] = {"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}}
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)

    assert [step.id for step in resolve_workbench(spec).pipeline] == [
        "ingest",
        "merge_map",
        "labels",
        "blank",
        "overflow",
        "ratio_yfp_cfp",
        "ratio_cfp_od600",
        "ratio_yfp_od600",
    ]


def test_plot_presets_expand_dual_reporter_screen_core(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [{"id": "ingest", "uses": "ingest/synergy_h1", "writes": {"df": "raw/df"}}]
    data["plots"] = {
        "presets": ["plots/plate_reader_dual_reporter_screen_core"],
        "defaults": {"reads": {"df": "raw/df"}},
        "specs": [],
    }
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)

    assert [item.id for item in resolve_workbench(spec).plots] == [
        "plot_time_series",
        "snapshot_bars_by_state",
        "ts_and_snap__yfp_over_cfp",
    ]


def test_validate_rejects_notebook_with_config_until_semantics_exist(tmp_path: Path) -> None:
    data = _base_config()
    data["notebooks"] = {"specs": [{"id": "eda", "uses": "notebook/eda", "with": {"theme": "lab"}}]}
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    with pytest.raises(ConfigError, match="notebook specs"):
        validate_job(spec, console=Console())
