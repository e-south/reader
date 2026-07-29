from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console
from rich.theme import Theme

from reader.contracts import builtin_contract_catalog
from reader.errors import ConfigError, ExecutionError, RecordError
from reader.protocols import ProtocolBinding, ProtocolSemanticProgram, builtin_protocol_catalog
from reader.runtime import ReaderRuntime
from reader.workbench import PluginSemantics
from reader.workbench.assets import AssetCatalog, build_plugin_asset
from reader.workbench.decl.model import (
    ExperimentDecl,
    NotebookDecl,
    PipelineDecl,
    PluginStepDecl,
    SurfaceDecl,
    WorkbenchDecl,
)
from reader.workbench.engine import run_spec
from reader.workbench.engine.invocations import (
    InvocationLedger,
    capture_revision_snapshot,
    declared_input_projection,
    produced_record_revisions,
)
from reader.workbench.experiment import AnnotationSemantics, ExperimentSemantics, OutputLayout, ResourceCatalog
from reader.workbench.graph import FileRef, PluginStep, RecordRef, ResourceRef
from reader.workbench.ports import dataframe_output
from reader.workbench.records import RecordStore
from reader.workbench.records.identity import BuildIdentity
from reader.workbench.registry import Plugin, PluginConfig, Registry


class _SyntheticIngestConfig(PluginConfig):
    pass


class _SyntheticIngest(Plugin):
    ConfigModel = _SyntheticIngestConfig

    @classmethod
    def input_ports(cls):
        return {}

    @classmethod
    def output_ports(cls):
        return {"df": dataframe_output("df", "tidy.v1")}

    def run(self, ctx, inputs, cfg):
        del ctx, inputs, cfg
        return {"df": pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})}


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_invocation_ledger_writes_attempt_and_terminal_result(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    ledger = InvocationLedger(experiment_root=tmp_path, outputs_dir=outputs)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest="sha256:source"),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[
            {
                "phase": "pipeline",
                "step_id": "ingest",
                "port": "raw",
                "ref": {"file": "inputs/raw.xlsx"},
            }
        ],
    )

    ledger.append_result(
        attempt,
        exit_status=0,
        produced_record_revisions=[
            {
                "record_id": "ingest/df",
                "revision": 1,
                "revision_digest": "sha256:revision",
            }
        ],
    )

    events = _events(outputs / "manifests" / "invocations.jsonl")
    assert [event["event"] for event in events] == ["attempt", "result"]
    assert events[0]["invocation_id"] == events[1]["invocation_id"] == attempt.invocation_id
    assert events[0]["config_digest"] == "sha256:config"
    assert events[0]["operation"] == "run"
    assert events[0]["build_identity"] == {
        "reader_version": "1.2.3",
        "source_digest": "sha256:source",
    }
    assert events[0]["selected_step_ids"] == {"pipeline": ["ingest"], "plots": [], "exports": []}
    assert events[0]["declared_inputs"] == [
        {
            "phase": "pipeline",
            "step_id": "ingest",
            "port": "raw",
            "ref": {"file": "inputs/raw.xlsx"},
        }
    ]
    assert events[1]["declared_inputs"] == events[0]["declared_inputs"]
    assert events[1]["exit_status"] == 0
    assert events[1]["status"] == "succeeded"
    assert events[1]["produced_record_revisions"] == [
        {
            "record_id": "ingest/df",
            "revision": 1,
            "revision_digest": "sha256:revision",
        }
    ]


def test_invocation_ledger_confines_outputs_to_experiment_root(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must stay under the experiment root"):
        InvocationLedger(experiment_root=tmp_path / "experiment", outputs_dir=tmp_path / "outside")


def test_invocation_failure_is_sanitized_and_bounded(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    ledger = InvocationLedger(experiment_root=tmp_path, outputs_dir=outputs)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest="sha256:source"),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    secret = "x" * 700

    ledger.append_result(
        attempt,
        exit_status=1,
        produced_record_revisions=[],
        failure=RuntimeError(f"failed at {tmp_path}/inputs/raw.xlsx token={secret}"),
    )

    result = _events(outputs / "manifests" / "invocations.jsonl")[1]
    failure = result["failure"]
    assert failure["type"] == "RuntimeError"
    assert str(tmp_path) not in failure["reason"]
    assert secret not in failure["reason"]
    assert "<redacted>" in failure["reason"]
    assert len(failure["reason"]) <= 500


def test_invocation_failure_redacts_common_credentials_without_losing_diagnostics(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    ledger = InvocationLedger(experiment_root=tmp_path, outputs_dir=outputs)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest="sha256:source"),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    failure = RuntimeError(
        "connection failed client_secret=hunter2 db_password='open sesame' "
        "service_token: abc123 Authorization: Basic dXNlcjpwYXNz "
        "secondary Authorization: Digest opaque-value "
        "at postgresql://reader:p%40ss@db.internal:5432/assays "
        "via https://api-user:api-pass@example.test/v1 "
        "callback=https://example.test/auth?refresh_token=query-secret&mode=read retry=3"
    )

    ledger.append_result(attempt, exit_status=1, produced_record_revisions=[], failure=failure)

    reason = _events(outputs / "manifests" / "invocations.jsonl")[1]["failure"]["reason"]
    for secret in (
        "hunter2",
        "open sesame",
        "abc123",
        "dXNlcjpwYXNz",
        "opaque-value",
        "reader:p%40ss",
        "api-user:api-pass",
        "query-secret",
    ):
        assert secret not in reason
    assert "client_secret=<redacted>" in reason
    assert "db_password=<redacted>" in reason
    assert "service_token=<redacted>" in reason
    assert "Authorization: Basic <redacted>" in reason
    assert "Authorization: Digest <redacted>" in reason
    assert "postgresql://<redacted>@db.internal:5432/assays" in reason
    assert "https://<redacted>@example.test/v1" in reason
    assert "refresh_token=<redacted>&mode=read" in reason
    assert "connection failed" in reason
    assert "retry=3" in reason


def test_invocation_failure_redacts_provider_credentials_and_preserves_status(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    ledger = InvocationLedger(experiment_root=tmp_path, outputs_dir=outputs)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest="sha256:source"),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    failure = RuntimeError(
        'provider failure "OPENAI_API_KEY": "sk-live" '
        "'AWS_SECRET_ACCESS_KEY'='aws-secret' X_AMZ_SIGNATURE : deadbeef "
        "API_KEY_VENDOR=vendor-key CLOUD_CREDENTIAL: cloud-credential "
        "SERVICE_AUTH_TOKEN='auth-token' signature mismatch stage=verify status=failed retry=3"
    )

    ledger.append_result(attempt, exit_status=1, produced_record_revisions=[], failure=failure)

    reason = _events(outputs / "manifests" / "invocations.jsonl")[1]["failure"]["reason"]
    for secret in (
        "sk-live",
        "aws-secret",
        "deadbeef",
        "vendor-key",
        "cloud-credential",
        "auth-token",
    ):
        assert secret not in reason
    for key in (
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "X_AMZ_SIGNATURE",
        "API_KEY_VENDOR",
        "CLOUD_CREDENTIAL",
        "SERVICE_AUTH_TOKEN",
    ):
        assert f"{key}=<redacted>" in reason
    assert "signature mismatch" in reason
    assert "stage=verify" in reason
    assert "status=failed" in reason
    assert "retry=3" in reason


def test_produced_record_revisions_reports_only_changed_revisions() -> None:
    unchanged = {"record_id": "existing/df", "revision": 1, "revision_digest": "sha256:existing"}
    before = {"existing/df": unchanged}
    after = {
        "existing/df": unchanged,
        "new/df": {"record_id": "new/df", "revision": 1, "revision_digest": "sha256:new"},
    }

    assert produced_record_revisions(before=before, after=after) == [after["new/df"]]


@pytest.mark.parametrize("corruption", ["empty_history", "divergent_final"])
def test_capture_revision_snapshot_rejects_malformed_catalog_lineage(tmp_path: Path, corruption: str) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synthetic",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    if corruption == "empty_history":
        catalog["history"]["ingest/df"] = []
    else:
        catalog["history"]["ingest/df"][-1]["config_digest"] = "sha256:divergent-history"
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="history"):
        capture_revision_snapshot(store)


def test_declared_input_projection_is_typed_relative_and_deterministic(tmp_path: Path) -> None:
    step = PluginStep(
        kind="pipeline",
        id="analyze",
        plugin="analysis/example",
        reads={
            "upstream": RecordRef("ingest/df"),
            "metadata": ResourceRef("sample_map", tmp_path / "inputs" / "metadata.xlsx"),
            "raw": FileRef(tmp_path / "inputs" / "raw.xlsx"),
        },
    )

    assert declared_input_projection(
        steps_by_phase={"pipeline": [step], "plots": [], "exports": []},
        experiment_root=tmp_path,
    ) == [
        {
            "phase": "pipeline",
            "step_id": "analyze",
            "port": "metadata",
            "ref": {"resource": "sample_map", "path": "inputs/metadata.xlsx"},
        },
        {
            "phase": "pipeline",
            "step_id": "analyze",
            "port": "raw",
            "ref": {"file": "inputs/raw.xlsx"},
        },
        {
            "phase": "pipeline",
            "step_id": "analyze",
            "port": "upstream",
            "ref": {"record": "ingest/df"},
        },
    ]


def test_declared_input_projection_rejects_paths_outside_experiment_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.xlsx"
    step = PluginStep(
        kind="pipeline",
        id="ingest",
        plugin="ingest/example",
        reads={"raw": FileRef(outside)},
    )

    with pytest.raises(ExecutionError, match="must stay under the experiment root"):
        declared_input_projection(
            steps_by_phase={"pipeline": [step], "plots": [], "exports": []},
            experiment_root=tmp_path,
        )


def test_run_spec_returns_exact_produced_record_revisions(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    decl = WorkbenchDecl(
        experiment=ExperimentDecl(id="synthetic", title="synthetic", lifecycle="active", root=tmp_path),
        experiment_semantics=ExperimentSemantics(
            protocol=ProtocolBinding(id="workbench/generic"),
            protocol_program=ProtocolSemanticProgram(protocol="workbench/generic"),
            annotations=AnnotationSemantics(),
            resources=ResourceCatalog(),
            layout=OutputLayout(
                outputs_dir=outputs,
                plots_subdir="plots",
                exports_subdir="exports",
                notebooks_subdir="notebooks",
            ),
        ),
        plotting_palette=None,
        pipeline=PipelineDecl(
            steps=(PluginStepDecl(id="ingest", plugin="ingest/synthetic"),),
        ),
        plots=SurfaceDecl(),
        exports=SurfaceDecl(),
        notebooks=NotebookDecl(),
    )
    registry = Registry(contracts=builtin_contract_catalog())
    registry.register(
        build_plugin_asset(
            plugin_id="ingest/synthetic",
            semantics=PluginSemantics(
                domain="generic",
                family="synthetic_ingest",
                summary="Synthetic ingest plugin for execution-result tests.",
            ),
            plugin_cls=_SyntheticIngest,
        )
    )
    runtime = ReaderRuntime(
        contracts=builtin_contract_catalog(),
        protocols=builtin_protocol_catalog(),
        plugins=registry,
        assets=AssetCatalog([]),
    )

    result = run_spec(
        decl,
        include_pipeline=True,
        include_plots=False,
        include_exports=False,
        log_level="ERROR",
        console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
        runtime=runtime,
    )

    assert result.invocation_id
    assert result.operation == "run"
    assert result.status == "succeeded"
    assert result.selected_steps.pipeline == ("ingest",)
    assert result.selected_steps.plots == ()
    assert result.selected_steps.exports == ()
    assert result.ledger_path == outputs / "manifests" / "invocations.jsonl"
    assert len(result.produced_record_revisions) == 1
    revision = result.produced_record_revisions[0]
    assert revision.record_id == "ingest/df"
    assert revision.revision == 1
    assert revision.revision_digest.startswith("sha256:")

    result_event = _events(result.ledger_path)[1]
    assert result_event["invocation_id"] == result.invocation_id
    assert result_event["produced_record_revisions"] == [revision.to_dict()]


def test_run_spec_rejects_symlinked_outputs_before_log_or_manifest_write(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outputs = experiment / "outputs"
    try:
        outputs.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    decl = WorkbenchDecl(
        experiment=ExperimentDecl(id="synthetic", title="synthetic", lifecycle="active", root=experiment),
        experiment_semantics=ExperimentSemantics(
            protocol=ProtocolBinding(id="workbench/generic"),
            protocol_program=ProtocolSemanticProgram(protocol="workbench/generic"),
            annotations=AnnotationSemantics(),
            resources=ResourceCatalog(),
            layout=OutputLayout(
                outputs_dir=outputs,
                plots_subdir="plots",
                exports_subdir="exports",
                notebooks_subdir="notebooks",
            ),
        ),
        plotting_palette=None,
        pipeline=PipelineDecl(),
        plots=SurfaceDecl(),
        exports=SurfaceDecl(),
        notebooks=NotebookDecl(),
    )

    with pytest.raises(ConfigError, match="outputs.*symlink|symlink.*outputs"):
        run_spec(
            decl,
            include_pipeline=False,
            include_plots=False,
            include_exports=False,
            log_level="ERROR",
            console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
        )

    assert list(outside.iterdir()) == []
