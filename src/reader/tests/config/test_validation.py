"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/config/test_validation.py

Validation tests for v4 config loading and read-contract checks.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from reader.core.errors import ConfigError
from reader.tests.support import base_reader_config, load_models, write_config
from reader.workbench import resolve_workbench
from reader.workbench.config import ReaderSpec
from reader.workbench.engine import validate as validate_job
from reader.workbench.graph import FileRef, RecordRef, ResourceRef


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
    with pytest.raises(ConfigError, match="reader/v4"):
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
    with pytest.raises(ConfigError, match="reader/v4"):
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


def test_load_rejects_legacy_semantics_section(tmp_path: Path) -> None:
    data = _base_config()
    data["semantics"] = {"groups": {"design_id": {"singletons": [{"A": ["a"]}]}}}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="Unsupported legacy/removed"):
        ReaderSpec.load(path)


def test_load_rejects_legacy_semantics_orderings(tmp_path: Path) -> None:
    data = _base_config()
    data["semantics"] = {"orderings": {"treatment": "bad"}}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="Unsupported legacy/removed"):
        ReaderSpec.load(path)


def test_load_normalizes_assay_orders_to_string_lists(tmp_path: Path) -> None:
    data = _base_config()
    data["assay"] = {"orders": {"states": {"column": "treatment_alias", "values": ["A", 2, True]}}}
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    assert spec.assay.orders["states"].column == "treatment_alias"
    assert spec.assay.orders["states"].values == ["A", "2", "True"]


def test_load_normalizes_assay_collections_to_string_lists(tmp_path: Path) -> None:
    data = _base_config()
    data["assay"] = {
        "collections": {
            "group_ab": {
                "column": "design_id",
                "items": {
                    "Group A": ["g1", 2],
                    "Group B": [True],
                },
            }
        }
    }
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    assert spec.assay.collections["group_ab"].column == "design_id"
    assert spec.assay.collections["group_ab"].items == {
        "Group A": ["g1", "2"],
        "Group B": ["True"],
    }


def test_load_normalizes_resource_paths(tmp_path: Path) -> None:
    data = _base_config()
    data["resources"] = {"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}}
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    assert spec.resources.by_id["sample_map"].kind == "file"
    assert spec.resources.by_id["sample_map"].path == "./inputs/metadata.xlsx"


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


def test_load_rejects_notebook_defaults(tmp_path: Path) -> None:
    data = _base_config()
    data["notebooks"] = {"defaults": {"with": {"theme": "x"}}, "specs": [{"id": "default", "template": "notebook/eda"}]}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="notebooks only supports specs"):
        ReaderSpec.load(path)


def test_load_rejects_notebook_overrides(tmp_path: Path) -> None:
    data = _base_config()
    data["notebooks"] = {
        "overrides": {"default": {"template": "notebook/basic"}},
        "specs": [{"id": "default", "template": "notebook/eda"}],
    }
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="notebooks only supports specs"):
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


def test_load_rejects_subdirs_that_escape_outputs(tmp_path: Path) -> None:
    data = _base_config()
    data["paths"]["notebooks"] = "../notebooks"
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="may not escape via '\\.\\.'"):
        ReaderSpec.load(path)


def test_load_rejects_outputs_relative_subdirs_with_parent_segments(tmp_path: Path) -> None:
    data = _base_config()
    data["paths"]["plots"] = "plots/../figures"
    path = write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="may not escape via '\\.\\.'"):
        ReaderSpec.load(path)


def test_workbench_resolves_file_reads_absolute_paths(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    data_path = tmp_path / "inputs.xlsx"
    data_path.write_text("stub", encoding="utf-8")
    data = _base_config()
    data["pipeline"]["steps"] = [
        {"id": "ingest", "plugin": "ingest/synergy_h1", "reads": {"raw": {"file": str(data_path)}}}
    ]
    path = write_config(exp_dir, data)
    _, decl = load_models(path)
    assert resolve_workbench(decl).pipeline[0].reads["raw"] == FileRef(path=data_path.resolve())


def test_workbench_resolves_resource_reads_to_files(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    data_path = exp_dir / "inputs.xlsx"
    data_path.write_text("stub", encoding="utf-8")
    data = _base_config()
    data["resources"] = {"sample_map": {"kind": "file", "path": "./inputs.xlsx"}}
    data["pipeline"]["steps"] = [
        {
            "id": "merge",
            "plugin": "transform/sample_map",
            "reads": {"df": "raw/df", "sample_map": {"resource": "sample_map"}},
        }
    ]
    path = write_config(exp_dir, data)
    _, decl = load_models(path)
    assert resolve_workbench(decl).pipeline[0].reads["sample_map"] == ResourceRef(
        resource_id="sample_map",
        path=data_path.resolve(),
    )


def test_workbench_requires_explicit_sample_map_resource(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    data = _base_config()
    data["pipeline"]["steps"] = [
        {
            "id": "merge",
            "plugin": "transform/sample_map",
            "reads": {"df": "raw/df", "sample_map": {"resource": "sample_map"}},
        }
    ]
    path = write_config(exp_dir, data)
    with pytest.raises(ConfigError, match="Unknown resource 'sample_map'"):
        load_models(path)


def test_workbench_requires_explicit_metadata_resource(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    data = _base_config()
    data["pipeline"]["steps"] = [
        {
            "id": "merge",
            "plugin": "transform/sample_metadata",
            "reads": {"df": "raw/df", "metadata": {"resource": "metadata"}},
        }
    ]
    path = write_config(exp_dir, data)
    with pytest.raises(ConfigError, match="Unknown resource 'metadata'"):
        load_models(path)


def test_validate_rejects_unexpected_reads(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [
        {"id": "ingest", "plugin": "ingest/synergy_h1"},
        {
            "id": "merge_map",
            "plugin": "transform/sample_map",
            "reads": {"df": "ingest/df", "plate_map": {"file": "./inputs/metadata.xlsx"}},
        },
    ]
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    with pytest.raises(ConfigError):
        validate_job(decl, console=Console())


def test_validate_rejects_unknown_read_labels(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [
        {"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "raw/df"}},
        {
            "id": "merge_map",
            "plugin": "transform/sample_map",
            "reads": {"df": "ingest/df", "sample_map": {"resource": "sample_map"}},
        },
    ]
    data["resources"] = {"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}}
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    with pytest.raises(ConfigError):
        validate_job(decl, console=Console())


def test_validate_rejects_duplicate_output_labels(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [
        {"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "shared/df"}},
        {
            "id": "overflow",
            "plugin": "transform/overflow_handling",
            "reads": {"df": "shared/df"},
            "writes": {"df": "shared/df"},
        },
    ]
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    with pytest.raises(ConfigError):
        validate_job(decl, console=Console())


def test_plot_defaults_apply_to_presets(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [{"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "raw/df"}}]
    data["plots"] = {
        "recipes": ["plots/plate_reader_yfp_time_series"],
        "defaults": {"reads": {"df": "raw/df"}},
        "specs": [],
    }
    data["exports"] = {"specs": []}
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    plot_specs = resolve_workbench(decl).plots
    assert plot_specs
    assert all(ps.reads.get("df") == RecordRef(record_id="raw/df") for ps in plot_specs)


def test_pipeline_presets_expand_dual_reporter_screen_base(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"] = {
        "recipes": ["plate_reader/synergy_h1", "plate_reader/dual_reporter_screen_base"],
        "steps": [],
    }
    data["resources"] = {"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}}
    path = write_config(tmp_path, data)
    _, decl = load_models(path)

    assert [step.id for step in resolve_workbench(decl).pipeline] == [
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
    data["pipeline"]["steps"] = [{"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "raw/df"}}]
    data["plots"] = {
        "recipes": ["plots/plate_reader_dual_reporter_screen_core"],
        "defaults": {"reads": {"df": "raw/df"}},
        "specs": [],
    }
    path = write_config(tmp_path, data)
    _, decl = load_models(path)

    assert [item.id for item in resolve_workbench(decl).plots] == [
        "plot_time_series",
        "snapshot_bars_by_state",
        "ts_and_snap__yfp_over_cfp",
    ]


def test_validate_accepts_partition_collection_ref(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [{"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "raw/df"}}]
    data["plots"] = {
        "defaults": {"reads": {"df": "raw/df"}},
        "specs": [
            {
                "id": "plot_time_series",
                "plugin": "plot/time_series",
                "with": {
                    "partition": {"collection_ref": "group_ab"},
                    "hue": "treatment",
                    "y": ["OD600"],
                },
            }
        ],
    }
    data["assay"] = {
        "collections": {
            "group_ab": {
                "column": "design_id",
                "items": {"Group A": ["g1", "g2"], "Group B": ["g3"]},
            }
        }
    }
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    validate_job(decl, console=Console())


def test_validate_rejects_unknown_partition_collection_ref(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [{"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "raw/df"}}]
    data["plots"] = {
        "defaults": {"reads": {"df": "raw/df"}},
        "specs": [
            {
                "id": "plot_time_series",
                "plugin": "plot/time_series",
                "with": {
                    "partition": {"collection_ref": "missing"},
                    "hue": "treatment",
                    "y": ["OD600"],
                },
            }
        ],
    }
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    with pytest.raises(ConfigError, match="collection_ref"):
        validate_job(decl, console=Console())


def test_validate_rejects_partition_collection_column_mismatch(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"]["steps"] = [{"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "raw/df"}}]
    data["plots"] = {
        "defaults": {"reads": {"df": "raw/df"}},
        "specs": [
            {
                "id": "plot_time_series",
                "plugin": "plot/time_series",
                "with": {
                    "partition": {"by": "design_id_alias", "collection_ref": "group_ab"},
                    "hue": "treatment",
                    "y": ["OD600"],
                },
            }
        ],
    }
    data["assay"] = {
        "collections": {
            "group_ab": {
                "column": "design_id",
                "items": {"Group A": ["g1"]},
            }
        }
    }
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    with pytest.raises(ConfigError, match="partition.collection_ref"):
        validate_job(decl, console=Console())


def test_load_rejects_notebook_with_config(tmp_path: Path) -> None:
    data = _base_config()
    data["notebooks"] = {"specs": [{"id": "eda", "template": "notebook/eda", "with": {"theme": "lab"}}]}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="notebooks.specs.0.with"):
        ReaderSpec.load(path)
