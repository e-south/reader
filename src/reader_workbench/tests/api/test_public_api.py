from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import reader_workbench as reader_package
from reader_workbench.api import (
    InspectionResult,
    NotebookResult,
    PlanResult,
    PluginCatalogResult,
    PluginDescriptorResult,
    RecordCatalogResult,
    RunResult,
    SurfaceCatalogResult,
    ValidationResult,
    VerificationResult,
    describe_plugin,
    inspect,
    notebook,
    open_experiment,
    plan,
    plots,
    plugins,
    records,
    run,
    validate,
    verify,
)
from reader_workbench.errors import ConfigError, RegistryError
from reader_workbench.runtime import ReaderRuntime, builtin_runtime
from reader_workbench.tests.support.configs import base_reader_config, write_config
from reader_workbench.workbench.config import ReaderSpec
from reader_workbench.workbench.decl import build_workbench_decl
from reader_workbench.workbench.engine.invocations import InvocationLedger
from reader_workbench.workbench.records import record_revision_digest
from reader_workbench.workbench.records.identity import current_build_identity
from reader_workbench.workbench.registry import Registry


def _generic_experiment(tmp_path: Path) -> Path:
    experiment_dir = tmp_path / "example"
    experiment_dir.mkdir()
    return write_config(experiment_dir, base_reader_config(experiment_id="example"))


def test_package_root_exposes_only_the_primary_experiment_entrypoint() -> None:
    assert reader_package.Experiment.__module__ == "reader_workbench.api.models"
    assert reader_package.open_experiment.__module__ == "reader_workbench.api.facade"


def test_package_root_missing_attribute_names_the_current_import_package() -> None:
    with pytest.raises(AttributeError, match="module 'reader_workbench' has no attribute 'missing'"):
        reader_package.__getattr__("missing")


def test_open_experiment_accepts_config_or_directory_without_creating_outputs(tmp_path: Path) -> None:
    config_path = _generic_experiment(tmp_path)

    from_config = open_experiment(config_path)
    from_directory = open_experiment(config_path.parent)

    assert from_config.config_path == config_path.resolve()
    assert from_directory.identity == from_config.identity
    assert from_config.identity.id == "example"
    assert from_config.identity.protocol == "workbench/generic"
    assert not hasattr(from_config, "spec")
    assert not hasattr(from_config, "declaration")
    assert not hasattr(from_config, "runtime")
    assert not (config_path.parent / "outputs").exists()


def test_experiment_read_surfaces_return_typed_results(tmp_path: Path) -> None:
    config_path = _generic_experiment(tmp_path)
    experiment = open_experiment(config_path)

    inspection = inspect(experiment)
    validation = validate(experiment, check_files=False)
    execution_plan = plan(experiment)
    plot_catalog = plots(experiment)
    record_catalog = records(experiment)
    verification = verify(experiment)

    assert isinstance(inspection, InspectionResult)
    assert inspection.experiment == experiment.identity
    assert inspection.implementation["readiness"]["state"] == "runnable"
    assert isinstance(validation, ValidationResult)
    assert validation.status == "ok"
    assert validation.check_files is False
    assert isinstance(execution_plan, PlanResult)
    assert execution_plan.plan["protocol"] == "workbench/generic"
    assert isinstance(plot_catalog, SurfaceCatalogResult)
    assert plot_catalog.kind == "plot"
    assert plot_catalog.entries == ()
    assert isinstance(record_catalog, RecordCatalogResult)
    assert record_catalog.catalog_exists is False
    assert record_catalog.catalog["schema_version"] is None
    assert record_catalog.catalog["provenance_epoch_id"] is None
    assert record_catalog.catalog["active_invocation_ledger"] is None
    assert record_catalog.entries == ()
    assert isinstance(verification, VerificationResult)
    assert verification.status == "failed"
    assert verification.issues[0]["code"] == "catalog.missing"
    assert not (config_path.parent / "outputs").exists()


def test_verify_ignores_records_retired_from_the_current_workbench(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.yaml",
        base_reader_config(
            experiment_id="example",
            protocol_id="plate_reader/dual_reporter_screen",
            protocol_analysis={"include_fold_change": False},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
            annotations={
                "labels": {
                    "design_id": {
                        "source": "design_id",
                        "output": "design_id_alias",
                        "values": {},
                    }
                }
            },
        ),
    )
    runtime = builtin_runtime()
    spec = ReaderSpec.load(config_path)
    declaration = build_workbench_decl(spec, source_path=config_path, protocols=runtime.protocols)
    experiment = open_experiment(config_path, runtime=runtime)
    layout = declaration.experiment_semantics.layout
    store = runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=declaration.experiment.root,
    )
    frame = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [1.0]})
    for producer_id, record_id, config_digest in (
        ("ingest", "ingest/df", declaration.config_digest),
        ("retired", "retired/df", declaration.config_digest),
    ):
        store.persist_dataframe(
            producer_id=producer_id,
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id=record_id,
            df=frame,
            contract_id="tidy.v1",
            inputs=[],
            config_digest=config_digest,
            producer_config_digest=f"sha256:{producer_id}",
        )
    ledger = InvocationLedger.for_store(store=store)
    attempt = ledger.append_attempt(
        config_digest=declaration.config_digest,
        build_identity=current_build_identity(),
        operation="run",
        selected_step_ids={"pipeline": ["ingest", "retired"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    revisions = [
        {
            "record_id": record.record_id,
            "revision": 1,
            "revision_digest": record_revision_digest(record, outputs_dir=store.root),
        }
        for record in store.iter_latest_records()
    ]
    ledger.append_result(attempt, exit_status=0, produced_record_revisions=revisions)

    current_catalog = records(experiment)
    historical_catalog = records(experiment, include_history=True)
    result = verify(experiment)

    assert [entry["record_id"] for entry in current_catalog.entries] == ["ingest/df"]
    assert {entry["record_id"] for entry in historical_catalog.entries} == {"ingest/df", "retired/df"}
    assert result.status == "ok"
    assert result.summary == {
        "checked": 1,
        "failed": 0,
        "unverifiable": 0,
        "invocations_checked": 1,
        "invocation_failures": 0,
    }
    assert [record["record_id"] for record in result.records] == ["ingest/df"]


def test_records_history_remains_readable_when_current_plugins_are_unavailable(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.yaml",
        base_reader_config(
            experiment_id="archived",
            protocol_id="logic/four_state_vector_collection",
        ),
    )
    builtin = builtin_runtime()
    archived_runtime = ReaderRuntime(
        contracts=builtin.contracts,
        protocols=builtin.protocols,
        plugins=Registry(contracts=builtin.contracts),
    )
    experiment = open_experiment(config_path, runtime=archived_runtime)
    store = archived_runtime.record_store(tmp_path / "outputs")
    store.persist_dataframe(
        producer_id="retired_collection",
        producer_plugin="transform/retired_collection",
        out_name="df",
        record_id="retired_collection/vectors",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:archived",
    )

    history = records(experiment, include_history=True)

    assert [record["record_id"] for record in history.entries] == ["retired_collection/vectors"]
    assert history.entries[0]["description"] == (
        "Description unavailable because plugin 'transform/retired_collection' is not registered."
    )
    with pytest.raises(RegistryError, match="Unknown plugin"):
        records(experiment)


def test_run_returns_typed_invocation_result_without_console_leakage(tmp_path: Path, capsys) -> None:
    config_path = _generic_experiment(tmp_path)
    experiment = open_experiment(config_path)

    result = run(experiment, log_level="ERROR")

    assert isinstance(result, RunResult)
    assert result.experiment == experiment.identity
    assert result.invocation_id
    assert result.provenance_epoch_id
    assert result.operation == "run"
    assert result.status == "succeeded"
    assert result.dry_run is False
    assert result.selected_steps.pipeline == ()
    assert result.selected_steps.plots == ()
    assert result.selected_steps.exports == ()
    assert result.produced_record_revisions == ()
    assert result.ledger_path == str(
        config_path.parent / "outputs" / "manifests" / "invocations" / f"{result.provenance_epoch_id}.jsonl"
    )
    assert result.to_dict()["selected_steps"] == {"pipeline": [], "plots": [], "exports": []}
    assert capsys.readouterr() == ("", "")

    events = [json.loads(line) for line in Path(result.ledger_path).read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["attempt", "result"]
    assert events[1]["invocation_id"] == result.invocation_id
    assert events[1]["produced_record_revisions"] == []
    catalog = records(experiment)
    assert catalog.catalog["schema_version"] == 4
    assert catalog.catalog["provenance_epoch_id"] == result.provenance_epoch_id
    assert catalog.catalog["active_invocation_ledger"] == result.ledger_path


def test_run_dry_run_returns_plan_without_writing_outputs(tmp_path: Path) -> None:
    config_path = _generic_experiment(tmp_path)
    experiment = open_experiment(config_path)

    result = run(experiment, dry_run=True)

    assert isinstance(result, RunResult)
    assert result.invocation_id is None
    assert result.provenance_epoch_id is None
    assert result.operation == "run"
    assert result.status == "planned"
    assert result.dry_run is True
    assert result.ledger_path is None
    assert result.produced_record_revisions == ()
    assert not (config_path.parent / "outputs").exists()


def test_run_reset_records_replaces_an_invalid_catalog_through_public_api(tmp_path: Path) -> None:
    config_path = _generic_experiment(tmp_path)
    experiment = open_experiment(config_path)
    records_path = config_path.parent / "outputs" / "manifests" / "records.json"
    records_path.parent.mkdir(parents=True)
    records_path.write_text(
        json.dumps({"schema_version": 3, "latest": {}, "history": {}}),
        encoding="utf-8",
    )

    result = run(experiment, reset_records=True)

    assert result.status == "succeeded"
    assert result.provenance_epoch_id
    catalog = records(experiment)
    assert catalog.catalog["schema_version"] == 4
    assert catalog.catalog["provenance_epoch_id"] == result.provenance_epoch_id
    assert catalog.catalog["active_invocation_ledger"] == result.ledger_path


@pytest.mark.parametrize(
    "run_options, expected",
    [
        ({"dry_run": True}, "reset_records cannot be combined with dry_run"),
        ({"only": "ingest"}, "reset_records requires a complete run"),
        ({"from_step": "ingest"}, "reset_records requires a complete run"),
        ({"until_step": "ingest"}, "reset_records requires a complete run"),
    ],
)
def test_run_reset_records_rejects_dry_or_partial_execution_before_mutation(
    tmp_path: Path,
    run_options: dict[str, object],
    expected: str,
) -> None:
    config_path = _generic_experiment(tmp_path)
    experiment = open_experiment(config_path)

    with pytest.raises(ConfigError, match=expected):
        run(experiment, reset_records=True, **run_options)

    assert not (config_path.parent / "outputs").exists()


def test_notebook_api_generates_protocol_owned_progressive_workbench(tmp_path: Path) -> None:
    config_path = _generic_experiment(tmp_path)
    experiment = open_experiment(config_path)

    result = notebook(experiment, name="review.py")

    assert isinstance(result, NotebookResult)
    assert result.created is True
    assert result.template == "notebook/eda"
    assert Path(result.path) == config_path.parent / "outputs" / "notebooks" / "review.py"
    body = Path(result.path).read_text(encoding="utf-8")
    assert "build_notebook_deliverable_selector" in body
    assert "render_notebook_deliverable_viewport" in body
    assert result.to_dict()["path"] == result.path


def test_notebook_api_does_not_expose_template_selection(tmp_path: Path) -> None:
    experiment = open_experiment(_generic_experiment(tmp_path))

    with pytest.raises(TypeError, match="unexpected keyword argument 'template'"):
        notebook(experiment, template="notebook/eda")  # type: ignore[call-arg]

    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("selector", ["from_step", "until_step", "only"])
def test_run_rejects_empty_step_selectors_without_writing_outputs(tmp_path: Path, selector: str) -> None:
    config_path = _generic_experiment(tmp_path)
    experiment = open_experiment(config_path)

    with pytest.raises(ConfigError, match=f"{selector} must be a non-empty step id"):
        run(experiment, dry_run=True, **{selector: "  "})

    assert not (config_path.parent / "outputs").exists()


def test_plugin_catalog_and_descriptor_expose_machine_readable_contracts() -> None:
    catalog = plugins(category="ingest", domain="cytometry")
    descriptor = describe_plugin("ingest/flow_cytometer")

    assert isinstance(catalog, PluginCatalogResult)
    assert [item.plugin for item in catalog.plugins] == ["ingest/flow_cytometer"]
    assert isinstance(descriptor, PluginDescriptorResult)
    assert descriptor.plugin.plugin == "ingest/flow_cytometer"
    assert descriptor.config_schema["additionalProperties"] is False
    assert {port.name: port.kind for port in descriptor.input_ports} == {"raw": "file_set"}
    assert {port.name: port.kind for port in descriptor.output_ports} == {
        "channels": "dataframe",
        "df": "dataframe",
    }
    dataframe_port = next(port for port in descriptor.output_ports if port.name == "df")
    assert dataframe_port.contract == "tidy.v1"
    assert dataframe_port.contract_surface is not None
    assert dataframe_port.contract_surface["minimum"] == "tidy.v1"


def test_plugin_catalog_can_be_limited_to_a_protocol_default_plan() -> None:
    catalog = plugins(category="transform", protocol="plate_reader/single_reporter_screen")

    assert catalog.plugins
    assert all(item.category == "transform" for item in catalog.plugins)
    assert {item.plugin for item in catalog.plugins} >= {
        "transform/blank_correction",
        "transform/ratio",
        "transform/sample_map",
    }


def test_describe_plugin_rejects_unknown_plugin() -> None:
    with pytest.raises(RegistryError, match="Unknown plugin"):
        describe_plugin("transform/not_registered")
