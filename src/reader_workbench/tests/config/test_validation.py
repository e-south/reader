from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from reader_workbench.errors import ConfigError
from reader_workbench.protocols.builtins import builtin_protocol_catalog
from reader_workbench.protocols.model import (
    BoundProtocol,
    CompiledProtocolPlan,
    ProtocolArtifactSpec,
    ProtocolBinding,
    ProtocolDescriptor,
    ProtocolExecutionPlan,
)
from reader_workbench.tests.support import base_reader_config, cytometry_test_gating_policy, load_models, write_config
from reader_workbench.workbench import resolve_workbench
from reader_workbench.workbench.config import ReaderSpec
from reader_workbench.workbench.decl.model import FileInputDecl, PluginStepDecl
from reader_workbench.workbench.engine import validate as validate_job
from reader_workbench.workbench.engine.validation import _configured_output_steps
from reader_workbench.workbench.experiment import ResourceCatalog
from reader_workbench.workbench.graph.normalize import normalize_input_binding


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


def test_load_reports_invalid_utf8_as_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ConfigError, match="Could not read UTF-8 config"):
        ReaderSpec.load(path)


def test_load_reports_config_io_failure_as_config_error(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("schema: reader/v8\n", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_config_read(candidate: Path, *args, **kwargs):
        if candidate == path:
            raise OSError("synthetic read failure")
        return original_read_text(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_config_read)

    with pytest.raises(ConfigError, match="Could not read config.*synthetic read failure"):
        ReaderSpec.load(path)


def test_load_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
schema: reader/v8
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
    with pytest.raises(ConfigError, match="reader/v8"):
        ReaderSpec.load(path)


def test_load_rejects_reader_v7_without_a_compatibility_path(tmp_path: Path) -> None:
    data = _base_config()
    data["schema"] = "reader/v7"
    path = write_config(tmp_path, data)

    with pytest.raises(
        ConfigError,
        match=r"Config schema must be 'reader/v8'\. This repo only supports reader/v8 \(found 'reader/v7'\)",
    ):
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
    with pytest.raises(ConfigError, match="reader/v8 rejects removed config keys"):
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
        protocol_id="logic/four_state_vector_screen",
        protocol_inputs={"state_map_ref": "induction_logic"},
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
            "ordered_state_spaces": {
                "induction_logic": {
                    "column": "treatment",
                    "state_order": ["00", "10", "01", "11"],
                    "values": {"00": "A", "10": "B", "01": "C", "11": "D"},
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


def test_validate_rejects_unsupported_config_in_dormant_plot_view(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_dormant_plot",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
        protocol_outputs={
            "plots": {
                "profile": "none",
                "views": {"raw_kinetics": {"unsupported_option": True}},
            }
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    _, decl = load_models(write_config(tmp_path, data))

    assert resolve_workbench(decl).plots == ()
    with pytest.raises(ConfigError, match="unsupported_option"):
        validate_job(decl, console=Console(record=True), check_files=False)


def test_validate_accepts_valid_config_in_dormant_plot_view(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_dormant_plot",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
        protocol_outputs={
            "plots": {
                "profile": "none",
                "views": {"raw_kinetics": {"y": ["OD600"]}},
            }
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    _, decl = load_models(write_config(tmp_path, data))

    assert resolve_workbench(decl).plots == ()
    summary = validate_job(decl, console=Console(record=True), check_files=False)

    assert summary["status"] == "ok"


def test_validate_rejects_invalid_semantic_reference_in_dormant_plot_view(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_dormant_plot",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
        protocol_outputs={
            "plots": {
                "profile": "none",
                "views": {"raw_kinetics": {"partition": {"collection_ref": "missing"}}},
            }
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    _, decl = load_models(write_config(tmp_path, data))

    assert resolve_workbench(decl).plots == ()
    with pytest.raises(ConfigError, match="collection_ref"):
        validate_job(decl, console=Console(record=True), check_files=False)


def test_validate_rejects_unsupported_config_in_excluded_export_artifact(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_dormant_export",
        protocol_id="cytometry/flow_panel",
        protocol_inputs={"gating": cytometry_test_gating_policy()},
        protocol_outputs={
            "exports": {
                "exclude": ["gate_definition_table"],
                "artifacts": {"gate_definition_table": {"unsupported_option": True}},
            }
        },
        resources={"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}},
    )
    _, decl = load_models(write_config(tmp_path, data))

    assert "gate_definition_table" not in {item.id for item in resolve_workbench(decl).exports}
    with pytest.raises(ConfigError, match="unsupported_option"):
        validate_job(decl, console=Console(record=True), check_files=False)


def test_validate_accepts_valid_config_in_excluded_export_artifact(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_dormant_export",
        protocol_id="cytometry/flow_panel",
        protocol_inputs={"gating": cytometry_test_gating_policy()},
        protocol_outputs={
            "exports": {
                "exclude": ["gate_definition_table"],
                "artifacts": {"gate_definition_table": {"index": True}},
            }
        },
        resources={"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}},
    )
    _, decl = load_models(write_config(tmp_path, data))

    assert "gate_definition_table" not in {item.id for item in resolve_workbench(decl).exports}
    summary = validate_job(decl, console=Console(record=True), check_files=False)

    assert summary["status"] == "ok"


def test_configured_output_validation_does_not_reactivate_unrelated_export_defaults() -> None:
    def _compile(protocol: BoundProtocol) -> CompiledProtocolPlan:
        selected = protocol.select_export_outputs(
            defaults=("default_export",),
            allowed={"configured_export", "default_export"},
        )
        if "default_export" in selected:
            raise ConfigError("unrelated default export was reactivated")
        return CompiledProtocolPlan(
            semantic_program=protocol.descriptor.semantic_program(),
            exports=tuple(
                PluginStepDecl(
                    id=output_id,
                    plugin="export/csv",
                    with_={"path": f"{output_id}.csv"},
                )
                for output_id in selected
            ),
        )

    descriptor = ProtocolDescriptor(
        protocol="test/provider_outputs",
        domain="generic",
        family="test",
        summary="Synthetic provider output-selection contract.",
        artifacts=(
            ProtocolArtifactSpec(id="configured_export", summary="Configured export."),
            ProtocolArtifactSpec(id="default_export", summary="Unrelated default export.", default=True),
        ),
        execution=ProtocolExecutionPlan(compiler=_compile),
    )
    protocol = BoundProtocol(
        descriptor=descriptor,
        outputs={
            "exports": {
                "exclude": ["default_export"],
                "artifacts": {"configured_export": {"index": True}},
            }
        },
    )

    assert protocol.compile().exports == ()
    _, configured_exports = _configured_output_steps(protocol=protocol)

    assert [step.id for step in configured_exports] == ["configured_export"]


def test_configured_output_validation_accepts_null_exclude_from_runtime_binding() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/dual_reporter_screen",
            analysis={"include_fold_change": False},
            outputs={
                "plots": {
                    "profile": "none",
                    "exclude": None,
                    "views": {"raw_kinetics": {"y": ["OD600"]}},
                }
            },
        )
    )

    assert protocol.compile().plots == ()
    configured_plots, _ = _configured_output_steps(protocol=protocol)

    assert [step.id for step in configured_plots] == ["raw_kinetics"]


def test_single_reporter_protocol_compiles_opt_in_subject_comparison(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_subject_comparison",
        protocol_id="plate_reader/single_reporter_screen",
        protocol_analysis={"reporter_channel": "RFP", "normalizer_channel": "OD600"},
        protocol_outputs={
            "plots": {
                "profile": "none",
                "include": ["subject_comparison"],
                "views": {
                    "subject_comparison": {
                        "ts_style": "assay_subject_id_alias",
                        "snap_x": "assay_subject_id_alias",
                        "snap_hue": "treatment_alias",
                        "snap_time": 14.0,
                    }
                },
            }
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    _, decl = load_models(write_config(tmp_path, data))
    plot = next(step for step in resolve_workbench(decl).plots if step.id == "subject_comparison")

    assert plot.plugin == "plot/ts_and_snap"
    assert plot.with_["ts_channel"] == "OD600"
    assert plot.with_["snap_channel"] == "RFP/OD600"


def test_cytometry_protocol_compiles_explicit_gating_into_normal_records_and_artifacts(tmp_path: Path) -> None:
    gating = cytometry_test_gating_policy()
    data = base_reader_config(
        experiment_id="exp_cytometry",
        protocol_id="cytometry/flow_panel",
        protocol_inputs={"gating": gating},
        resources={"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}},
    )
    spec, decl = load_models(write_config(tmp_path, data))
    workbench = resolve_workbench(decl)
    step = next(item for item in workbench.pipeline if item.id == "cytometry_gating")
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id=spec.protocol.id,
            inputs=spec.protocol.inputs,
            analysis=spec.protocol.analysis,
            outputs=spec.protocol.outputs.model_dump(exclude_none=True),
        )
    )

    assert step.plugin == "transform/cytometry_gating"
    assert step.with_ == {}
    assert protocol.effective_plugin_config(plugin_id=step.plugin, step_with=step.with_) == gating
    assert {name: ref.record_id for name, ref in step.writes.items()} == {
        "gate_definition": "cytometry_gating/gate_definition",
        "gated_events": "cytometry_gating/gated_events",
        "sample_stats": "cytometry_gating/sample_stats",
        "group_stats": "cytometry_gating/group_stats",
        "qc": "cytometry_gating/qc",
    }
    assert [(item.id, item.plugin) for item in workbench.plots] == [("gating_diagnostic", "plot/cytometry_diagnostic")]
    assert [item.id for item in workbench.exports] == [
        "gate_definition_table",
        "sample_stats_table",
        "group_stats_table",
        "qc_table",
    ]


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


def test_load_normalizes_record_resource_identity(tmp_path: Path) -> None:
    data = _base_config()
    data["resources"] = {
        "source": {
            "kind": "record",
            "experiment": " source-experiment ",
            "record": " annotated/df ",
        }
    }

    spec = ReaderSpec.load(write_config(tmp_path, data))

    source = spec.resources.by_id["source"]
    assert source.kind == "record"
    assert source.experiment == "source-experiment"
    assert source.record == "annotated/df"


def test_load_rejects_file_fields_on_record_resource(tmp_path: Path) -> None:
    data = _base_config()
    data["resources"] = {
        "source": {
            "kind": "record",
            "experiment": "source-experiment",
            "record": "annotated/df",
            "path": "./outputs/table.parquet",
        }
    }

    with pytest.raises(ConfigError, match=r"resources\.source has unknown keys \['path'\]"):
        ReaderSpec.load(write_config(tmp_path, data))


def test_load_rejects_non_list_annotation_collection_items(tmp_path: Path) -> None:
    data = _base_config()
    data["annotations"] = {
        "collections": {"group_ab": {"column": "design_id", "items": {"A": "g1"}}},
    }
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="annotations.collections.group_ab.items entries must be lists"):
        ReaderSpec.load(path)


def test_load_rejects_removed_notebook_output_selection(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_cyto",
        protocol_id="cytometry/flow_panel",
        protocol_inputs={"gating": cytometry_test_gating_policy()},
    )
    data["resources"] = {"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}}
    data["protocol"]["outputs"] = {"notebook": {"template": "notebook/missing"}}
    path = write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="protocol.outputs"):
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
        protocol_id="logic/four_state_vector_screen",
        protocol_inputs={"state_map_ref": "induction_logic"},
        protocol_analysis={"include_four_state_vector": False, "include_fold_change": False},
        protocol_outputs={
            "plots": {"profile": "none", "include": ["logic_symmetry"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "ordered_state_spaces": {
                "induction_logic": {
                    "column": "treatment",
                    "state_order": ["00", "10", "01", "11"],
                    "values": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    workbench = resolve_workbench(decl)
    assert "promote_to_tidy_plus_map" in [step.id for step in workbench.pipeline]
    assert "logic_symmetry_summary" in [step.id for step in workbench.pipeline]
    assert "four_state_vector" not in [step.id for step in workbench.pipeline]
    assert [step.id for step in workbench.plots] == ["logic_symmetry"]
    assert workbench.plots[0].reads["table"].record_id == "logic_symmetry/table"
    assert workbench.exports == ()


def test_four_state_vector_default_plot_profile_respects_vector_opt_out(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_logic",
        protocol_id="logic/four_state_vector_screen",
        protocol_inputs={"state_map_ref": "induction_logic"},
        protocol_analysis={"include_four_state_vector": False, "include_fold_change": False},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "ordered_state_spaces": {
                "induction_logic": {
                    "column": "treatment",
                    "state_order": ["00", "10", "01", "11"],
                    "values": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    workbench = resolve_workbench(decl)

    assert "four_state_vector" not in [step.id for step in workbench.pipeline]
    assert "four_state_vector_heatmap" not in [plot.id for plot in workbench.plots]
    assert [plot.id for plot in workbench.plots] == ["raw_kinetics"]


def test_four_state_vector_explicit_workbook_export_compiles_vector_when_analysis_opts_out(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_logic",
        protocol_id="logic/four_state_vector_screen",
        protocol_inputs={"state_map_ref": "induction_logic"},
        protocol_analysis={"include_four_state_vector": False, "include_fold_change": False},
        protocol_outputs={
            "plots": {"profile": "none"},
            "exports": {"include": ["logic_summary_workbook"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "ordered_state_spaces": {
                "induction_logic": {
                    "column": "treatment",
                    "state_order": ["00", "10", "01", "11"],
                    "values": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)

    workbench = resolve_workbench(decl)

    assert [step.id for step in workbench.pipeline][-2:] == ["promote_to_tidy_plus_map", "four_state_vector"]
    assert [export.id for export in workbench.exports] == ["logic_summary_workbook"]
    assert workbench.exports[0].reads["df"].record_id == "four_state_vector/vector"


def test_four_state_vector_delta_reaches_transform_config(tmp_path: Path) -> None:
    data = base_reader_config(
        experiment_id="exp_logic",
        protocol_id="logic/four_state_vector_screen",
        protocol_inputs={"state_map_ref": "induction_logic"},
        protocol_analysis={
            "include_four_state_vector": True,
            "include_fold_change": False,
            "four_state_vector": {
                "intensity_log2_offset_delta": 0.25,
            },
        },
        protocol_outputs={"plots": {"profile": "none"}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "ordered_state_spaces": {
                "induction_logic": {
                    "column": "treatment",
                    "state_order": ["00", "10", "01", "11"],
                    "values": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    path = write_config(tmp_path, data)
    _, decl = load_models(path)
    workbench = resolve_workbench(decl)

    vector_step = next(step for step in workbench.pipeline if step.id == "four_state_vector")
    assert vector_step.with_["log2_offset_delta"] == pytest.approx(0.25)


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
