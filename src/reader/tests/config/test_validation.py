from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from reader.errors import ConfigError
from reader.tests.support import base_reader_config, load_models, write_config
from reader.workbench import resolve_workbench
from reader.workbench.config import ReaderSpec
from reader.workbench.decl.model import FileInputDecl
from reader.workbench.engine import validate as validate_job
from reader.workbench.experiment import ResourceCatalog
from reader.workbench.graph.normalize import normalize_input_binding


def _base_config() -> dict:
    return base_reader_config(experiment_id="exp_001", title="Example")


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as err:
        pytest.skip(f"symlinks unavailable: {err}")


def test_load_rejects_non_mapping(tmp_path: Path) -> None:
    path = write_config(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        ReaderSpec.load(path)


def test_load_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
schema: reader/v7
experiment:
  id: first
  id: second
protocol:
  id: workbench/generic
""",
    )

    with pytest.raises(ConfigError, match="duplicate key 'id'"):
        ReaderSpec.load(path)


def test_load_requires_schema_marker(tmp_path: Path) -> None:
    data = _base_config()
    data.pop("schema")
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="reader/v7"):
        ReaderSpec.load(path)


def test_load_requires_explicit_experiment_id(tmp_path: Path) -> None:
    exp_dir = tmp_path / "20269999_example_exp"
    exp_dir.mkdir()
    data = _base_config()
    data["experiment"] = {}
    path = write_config(exp_dir, data)
    with pytest.raises(ConfigError, match="experiment.id is required"):
        ReaderSpec.load(path)


def test_load_derives_title_from_id_when_missing(tmp_path: Path) -> None:
    data = _base_config()
    data["experiment"] = {"id": "exp_alpha"}
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    assert spec.experiment.id == "exp_alpha"
    assert spec.experiment.title == "exp_alpha"
    assert spec.experiment.lifecycle == "active"


def test_load_accepts_explicit_experiment_lifecycle(tmp_path: Path) -> None:
    data = _base_config()
    data["experiment"]["lifecycle"] = "draft"
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    assert spec.experiment.lifecycle == "draft"


def test_load_rejects_unknown_experiment_lifecycle(tmp_path: Path) -> None:
    data = _base_config()
    data["experiment"]["lifecycle"] = "unsupported"
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="experiment.lifecycle must be one of"):
        ReaderSpec.load(path)


def test_load_rejects_outputs_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "escaped_outputs"
    outside.mkdir(parents=True, exist_ok=True)
    _symlink_or_skip(tmp_path / "outputs_link", outside)

    data = _base_config()
    data["paths"]["outputs"] = "./outputs_link"
    path = write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="paths.outputs must stay under the experiment root"):
        load_models(path)


def test_load_rejects_resource_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_resource.xlsx"
    outside.write_text("x", encoding="utf-8")
    _symlink_or_skip(tmp_path / "metadata_link.xlsx", outside)

    data = _base_config()
    data["resources"] = {"sample_map": {"kind": "file", "path": "./metadata_link.xlsx"}}
    path = write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="resources.sample_map.path must stay under the experiment root"):
        load_models(path)


def test_normalize_input_binding_rejects_file_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.csv"
    outside.write_text("value\n1\n", encoding="utf-8")
    _symlink_or_skip(tmp_path / "data_link.csv", outside)

    with pytest.raises(ConfigError, match="must stay under the experiment root"):
        normalize_input_binding(
            FileInputDecl(path="./data_link.csv"),
            root=tmp_path,
            resources=ResourceCatalog(),
            section="pipeline",
            step_id="ingest",
            key="raw",
        )


def test_load_rejects_removed_workflow_sections(tmp_path: Path) -> None:
    data = _base_config()
    data["pipeline"] = {"steps": []}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="reader/v7 rejects removed config keys"):
        ReaderSpec.load(path)


def test_load_rejects_removed_protocol_with_key(tmp_path: Path) -> None:
    data = _base_config()
    data["protocol"] = {"id": "workbench/generic", "with": {"x": 1}}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="protocol keys"):
        ReaderSpec.load(path)


def test_load_rejects_unknown_protocol_top_level_key(tmp_path: Path) -> None:
    data = _base_config()
    data["protocol"]["unexpected"] = {"x": 1}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="protocol has unknown keys"):
        ReaderSpec.load(path)


def test_load_rejects_unknown_protocol_input_key_for_bound_protocol(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_plate",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}, "unknown_block": {"x": 1}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="protocol.inputs for 'plate_reader/dual_reporter_screen' has unknown keys"):
        load_models(path)


def test_load_rejects_unknown_protocol_analysis_key_for_bound_protocol(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_plate",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": True, "include_fold_changee": False},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="protocol.analysis for 'plate_reader/dual_reporter_screen' has unknown keys"):
        load_models(path)


def test_load_accepts_extended_overflow_policy_keys(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_logic",
        protocol_id="logic/sfxi_screen",
        protocol_inputs={"logic_map_ref": "induction_logic"},
        protocol_analysis={
            "preprocessing": {
                "overflow": {
                    "action": "max",
                    "clip_quantile": 0.995,
                    "cap_strategy": "quantile",
                    "flag_column": "overflow",
                    "treat_inf_as_overflow": True,
                }
            }
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "logic_maps": {
                "induction_logic": {
                    "column": "treatment",
                    "corners": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    overflow = next(step for step in resolve_workbench(decl).pipeline if step.id == "overflow")
    assert overflow.with_["cap_strategy"] == "quantile"
    assert overflow.with_["flag_column"] == "overflow"


def test_load_rejects_unknown_protocol_outputs_key(tmp_path: Path) -> None:
    data = _base_config()
    data["protocol"]["outputs"] = {"plots": {"profile": "none", "include": ["generic_qc"], "unexpected": True}}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="protocol.outputs.plots has unknown keys"):
        ReaderSpec.load(path)


def test_load_normalizes_annotations_and_resources(tmp_path: Path) -> None:
    data = _base_config()
    data["resources"] = {"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}}
    data["annotations"] = {
        "orders": {"states": {"column": "treatment_alias", "values": ["A", 2, True]}},
        "collections": {"group_ab": {"column": "design_id", "items": {"A": ["g1", 2], "B": [True]}}},
    }
    path = write_config(tmp_path, data)
    spec = ReaderSpec.load(path)
    assert spec.resources.by_id["sample_map"].path == "./inputs/metadata.xlsx"
    assert spec.annotations.orders["states"].values == ["A", "2", "True"]
    assert spec.annotations.collections["group_ab"].items == {"A": ["g1", "2"], "B": ["True"]}


def test_load_rejects_non_list_annotation_collection_items(tmp_path: Path) -> None:
    data = _base_config()
    data["annotations"] = {
        "collections": {"group_ab": {"column": "design_id", "items": {"A": "g1"}}},
    }
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="annotations.collections.group_ab.items entries must be lists"):
        ReaderSpec.load(path)


def test_validate_rejects_notebook_template_outside_protocol_policy(tmp_path: Path) -> None:
    data = base_reader_config(experiment_id="exp_cyto", protocol_id="cytometry/flow_panel")
    data["resources"] = {"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}}
    data["protocol"]["outputs"] = {"notebook": {"template": "notebook/eda"}}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="does not allow notebook template"):
        load_models(path)


def test_validate_rejects_missing_required_plate_reader_resource(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_plate",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    with pytest.raises(ConfigError, match="unknown resource 'sample_map'"):
        validate_job(decl, console=Console(record=True), check_files=False)


def test_protocol_compiler_expands_plate_reader_pipeline(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_plate",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
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


def test_protocol_analysis_and_outputs_adjust_compiled_protocol(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_logic",
        protocol_id="logic/sfxi_screen",
        protocol_inputs={"logic_map_ref": "induction_logic"},
        protocol_analysis={"include_vec8": False, "include_fold_change": False},
        protocol_outputs={
            "plots": {"profile": "none", "include": ["logic_symmetry"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "logic_maps": {
                "induction_logic": {
                    "column": "treatment",
                    "corners": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    workbench = resolve_workbench(decl)
    assert "promote_to_tidy_plus_map" in [step.id for step in workbench.pipeline]
    assert "sfxi_vec8" not in [step.id for step in workbench.pipeline]
    assert [step.id for step in workbench.plots] == ["logic_symmetry"]
    assert workbench.exports == ()


def test_sfxi_default_plot_profile_respects_vec8_opt_out(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_logic",
        protocol_id="logic/sfxi_screen",
        protocol_inputs={"logic_map_ref": "induction_logic"},
        protocol_analysis={"include_vec8": False, "include_fold_change": False},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "logic_maps": {
                "induction_logic": {
                    "column": "treatment",
                    "corners": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    workbench = resolve_workbench(decl)

    assert "sfxi_vec8" not in [step.id for step in workbench.pipeline]
    assert "sfxi_vec8_heatmap" not in [plot.id for plot in workbench.plots]
    assert [plot.id for plot in workbench.plots] == [
        "raw_kinetics",
        "endpoint_by_condition",
        "endpoint_by_design",
        "intensity_overview",
    ]


def test_sfxi_objective_delta_reaches_vec8_and_setpoint_plot_configs(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_logic",
        protocol_id="logic/sfxi_screen",
        protocol_inputs={"logic_map_ref": "induction_logic"},
        protocol_analysis={
            "include_vec8": True,
            "include_fold_change": False,
            "sfxi_objective": {
                "intensity_log2_offset_delta": 0.25,
                "setpoints": {"and": [0.0, 0.0, 0.0, 1.0]},
                "scaling": {"percentile": 95, "min_n": 1, "eps": 1.0e-8},
            },
        },
        protocol_outputs={"plots": {"profile": "none", "include": ["sfxi_setpoint_scatter"]}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "logic_maps": {
                "induction_logic": {
                    "column": "treatment",
                    "corners": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    workbench = resolve_workbench(decl)

    vec8_step = next(step for step in workbench.pipeline if step.id == "sfxi_vec8")
    scatter_plot = next(step for step in workbench.plots if step.id == "sfxi_setpoint_scatter")

    assert vec8_step.with_["log2_offset_delta"] == pytest.approx(0.25)
    assert scatter_plot.with_["intensity_log2_offset_delta"] == pytest.approx(0.25)


def test_validate_accepts_partition_collection_ref(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_partition",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
        protocol_outputs={
            "plots": {
                "profile": "none",
                "include": ["raw_kinetics"],
                "views": {
                    "raw_kinetics": {
                        "partition": {"collection_ref": "group_ab"},
                        "hue": "treatment",
                        "y": ["OD600"],
                    }
                },
            }
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={"collections": {"group_ab": {"column": "design_id", "items": {"A": ["g1", "g2"], "B": ["g3"]}}}},
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    validate_job(decl, console=Console())


def test_validate_rejects_unknown_partition_collection_ref(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_partition",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
        protocol_outputs={
            "plots": {
                "profile": "none",
                "include": ["raw_kinetics"],
                "views": {
                    "raw_kinetics": {
                        "partition": {"collection_ref": "missing"},
                        "hue": "treatment",
                        "y": ["OD600"],
                    }
                },
            }
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    with pytest.raises(ConfigError, match="collection_ref"):
        validate_job(decl, console=Console())
