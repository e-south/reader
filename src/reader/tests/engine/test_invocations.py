from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from filelock import Timeout
from rich.console import Console
from rich.theme import Theme

from reader.contracts import builtin_contract_catalog
from reader.errors import ConfigError, ExecutionError, InvocationFinalizationError, RecordError
from reader.protocols import ProtocolBinding, ProtocolSemanticProgram, builtin_protocol_catalog
from reader.runtime import ReaderRuntime
from reader.workbench import PluginSemantics
from reader.workbench.assets import build_plugin_asset
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
from reader.workbench.records import RecordStore, verify_record_store
from reader.workbench.records.identity import BuildIdentity
from reader.workbench.registry import Plugin, PluginConfig, Registry

_SOURCE_DIGEST = "sha256:" + ("a" * 64)


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


class _TrackingLock:
    def __init__(
        self,
        *,
        failure: BaseException | None = None,
        release_failure: BaseException | None = None,
    ) -> None:
        self.depth = 0
        self.failure = failure
        self.release_failure = release_failure

    def acquire(self, *_args, **_kwargs):
        if self.failure is not None:
            raise self.failure
        self.depth += 1
        return self

    def release(self, *_args, **_kwargs) -> None:
        self.depth -= 1
        if self.depth == 0 and self.release_failure is not None:
            raise self.release_failure

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_args) -> None:
        self.release()


def _synthetic_decl_and_runtime(tmp_path: Path) -> tuple[WorkbenchDecl, ReaderRuntime]:
    decl = WorkbenchDecl(
        experiment=ExperimentDecl(id="synthetic", title="synthetic", lifecycle="active", root=tmp_path),
        experiment_semantics=ExperimentSemantics(
            protocol=ProtocolBinding(id="workbench/generic"),
            protocol_program=ProtocolSemanticProgram(protocol="workbench/generic"),
            annotations=AnnotationSemantics(),
            resources=ResourceCatalog(),
            layout=OutputLayout(
                outputs_dir=tmp_path / "outputs",
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
    )
    return decl, runtime


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _direct_ledger(*, experiment_root: Path, outputs_dir: Path) -> InvocationLedger:
    return InvocationLedger(
        experiment_root=experiment_root,
        outputs_dir=outputs_dir,
        provenance_epoch_id=str(uuid4()),
    )


def test_invocation_ledger_writes_attempt_and_terminal_result(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    epoch_id = str(uuid4())
    ledger = InvocationLedger(
        experiment_root=tmp_path,
        outputs_dir=outputs,
        provenance_epoch_id=epoch_id,
    )
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
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

    events = _events(outputs / "manifests" / "invocations" / f"{epoch_id}.jsonl")
    assert [event["event"] for event in events] == ["attempt", "result"]
    assert events[0]["invocation_id"] == events[1]["invocation_id"] == attempt.invocation_id
    assert events[0]["schema"] == events[1]["schema"] == "reader.invocation/v2"
    assert events[0]["provenance_epoch_id"] == events[1]["provenance_epoch_id"] == epoch_id
    assert events[0]["config_digest"] == "sha256:config"
    assert events[0]["operation"] == "run"
    assert events[0]["build_identity"] == {
        "reader_version": "1.2.3",
        "source_digest": _SOURCE_DIGEST,
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


def test_invocation_ledger_restores_previous_boundary_after_short_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    ledger = _direct_ledger(experiment_root=tmp_path, outputs_dir=outputs)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    before = ledger.path.read_bytes()
    original_write = os.write

    def _short_write(descriptor: int, payload: bytes) -> int:
        partial = payload[: max(1, len(payload) // 2)]
        return original_write(descriptor, partial)

    monkeypatch.setattr(os, "write", _short_write)

    with pytest.raises(ExecutionError, match="append was incomplete"):
        ledger.append_result(attempt, exit_status=0, produced_record_revisions=[])

    assert ledger.path.read_bytes() == before
    assert [event["event"] for event in _events(ledger.path)] == ["attempt"]


def test_invocation_ledger_restores_previous_boundary_after_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    ledger = _direct_ledger(experiment_root=tmp_path, outputs_dir=outputs)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    before = ledger.path.read_bytes()
    original_fsync = os.fsync
    calls = 0

    def _fail_once(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fsync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", _fail_once)

    with pytest.raises(ExecutionError, match="Could not append invocation event"):
        ledger.append_result(attempt, exit_status=0, produced_record_revisions=[])

    assert ledger.path.read_bytes() == before
    assert [event["event"] for event in _events(ledger.path)] == ["attempt"]


def test_invocation_ledger_confines_outputs_to_experiment_root(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must stay under the experiment root"):
        _direct_ledger(experiment_root=tmp_path / "experiment", outputs_dir=tmp_path / "outside")


@pytest.mark.parametrize("target_scope", ["inside", "outside"])
def test_invocation_ledger_append_rejects_symlinked_manifest_parent(
    tmp_path: Path,
    target_scope: str,
) -> None:
    experiment = tmp_path / "experiment"
    outputs = experiment / "outputs"
    outputs.mkdir(parents=True)
    target = experiment / "redirected-manifests" if target_scope == "inside" else tmp_path / "outside"
    target.mkdir()
    try:
        (outputs / "manifests").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    ledger = _direct_ledger(experiment_root=experiment, outputs_dir=outputs)

    with pytest.raises(ConfigError, match="manifest directory must not be a symlink"):
        ledger.append_attempt(
            config_digest="sha256:config",
            build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
            operation="run",
            selected_step_ids={"pipeline": [], "plots": [], "exports": []},
            declared_inputs=[],
        )

    assert list(target.iterdir()) == []


def test_invocation_ledger_append_rejects_symlinked_ledger(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "outputs" / "manifests" / "invocations"
    ledger_dir.mkdir(parents=True)
    target = tmp_path / "outside.jsonl"
    target.write_text("outside evidence\n", encoding="utf-8")
    ledger = _direct_ledger(experiment_root=tmp_path, outputs_dir=tmp_path / "outputs")
    try:
        ledger.path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ConfigError, match="ledger must not be a symlink"):
        ledger.append_attempt(
            config_digest="sha256:config",
            build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
            operation="run",
            selected_step_ids={"pipeline": [], "plots": [], "exports": []},
            declared_inputs=[],
        )

    assert target.read_text(encoding="utf-8") == "outside evidence\n"


def test_invocation_ledger_append_rejects_parent_swap_after_construction(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    outputs = experiment / "outputs"
    outputs.mkdir(parents=True)
    ledger = _direct_ledger(experiment_root=experiment, outputs_dir=outputs)
    outside = tmp_path / "outside"
    manifests = outside / "manifests"
    manifests.mkdir(parents=True)
    outputs.rmdir()
    try:
        outputs.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ConfigError, match="outputs sink root must not use symlink"):
        ledger.append_attempt(
            config_digest="sha256:config",
            build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
            operation="run",
            selected_step_ids={"pipeline": [], "plots": [], "exports": []},
            declared_inputs=[],
        )

    assert list(manifests.iterdir()) == []


def test_invocation_ledger_rechecks_manifest_parent_after_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    ledger = _direct_ledger(experiment_root=tmp_path, outputs_dir=outputs)
    original_mkdir = Path.mkdir

    def _swap_parent(path: Path, *args, **kwargs) -> None:
        if path == ledger.path.parent:
            path.symlink_to(outside, target_is_directory=True)
            return
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _swap_parent)

    with pytest.raises(ConfigError, match="ledger directory must not be a symlink"):
        ledger.append_attempt(
            config_digest="sha256:config",
            build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
            operation="run",
            selected_step_ids={"pipeline": [], "plots": [], "exports": []},
            declared_inputs=[],
        )

    assert list(outside.iterdir()) == []


def test_invocation_failure_is_sanitized_and_bounded(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    ledger = _direct_ledger(experiment_root=tmp_path, outputs_dir=outputs)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
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

    result = _events(ledger.path)[1]
    failure = result["failure"]
    assert failure["type"] == "RuntimeError"
    assert str(tmp_path) not in failure["reason"]
    assert secret not in failure["reason"]
    assert "<redacted>" in failure["reason"]
    assert len(failure["reason"]) <= 500


def test_invocation_failure_redacts_common_credentials_without_losing_diagnostics(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    ledger = _direct_ledger(experiment_root=tmp_path, outputs_dir=outputs)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    failure = RuntimeError(
        "connection failed client_secret=hunter2 db_password='open sesame' "  # pragma: allowlist secret
        "service_token: abc123 Authorization: Basic dXNlcjpwYXNz "
        "secondary Authorization: Digest opaque-value "
        "at postgresql://reader:p%40ss@db.internal:5432/assays "  # pragma: allowlist secret
        "via https://api-user:api-pass@example.test/v1 "  # pragma: allowlist secret
        "callback=https://example.test/auth?refresh_token=query-secret&mode=read retry=3"
    )

    ledger.append_result(attempt, exit_status=1, produced_record_revisions=[], failure=failure)

    reason = _events(ledger.path)[1]["failure"]["reason"]
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
    ledger = _direct_ledger(experiment_root=tmp_path, outputs_dir=outputs)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    failure = RuntimeError(
        'provider failure "OPENAI_API_KEY": "sk-live" '  # pragma: allowlist secret
        "'AWS_SECRET_ACCESS_KEY'='aws-secret' X_AMZ_SIGNATURE : deadbeef "  # pragma: allowlist secret
        "API_KEY_VENDOR=vendor-key CLOUD_CREDENTIAL: cloud-credential "
        "SERVICE_AUTH_TOKEN='auth-token' signature mismatch stage=verify status=failed retry=3"
    )

    ledger.append_result(attempt, exit_status=1, produced_record_revisions=[], failure=failure)

    reason = _events(ledger.path)[1]["failure"]["reason"]
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
    decl, runtime = _synthetic_decl_and_runtime(tmp_path)

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
    store = runtime.record_store(outputs, experiment_root=tmp_path, create=False)
    assert result.ledger_path == store.invocation_ledger_path()
    assert len(result.produced_record_revisions) == 1
    revision = result.produced_record_revisions[0]
    assert revision.record_id == "ingest/df"
    assert revision.revision == 1
    assert revision.revision_digest.startswith("sha256:")

    result_event = _events(result.ledger_path)[1]
    assert result_event["invocation_id"] == result.invocation_id
    assert result_event["produced_record_revisions"] == [revision.to_dict()]


def test_run_spec_holds_one_writer_lease_across_plugin_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decl, runtime = _synthetic_decl_and_runtime(tmp_path)
    lease = _TrackingLock()
    observed_depths: list[int] = []
    original_run = _SyntheticIngest.run

    def _run_under_observation(self, ctx, inputs, cfg):
        observed_depths.append(lease.depth)
        return original_run(self, ctx, inputs, cfg)

    monkeypatch.setattr(RecordStore, "provenance_lock", property(lambda _store: lease))
    monkeypatch.setattr(_SyntheticIngest, "run", _run_under_observation)

    run_spec(
        decl,
        include_pipeline=True,
        include_plots=False,
        include_exports=False,
        log_level="ERROR",
        console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
        runtime=runtime,
    )

    assert observed_depths == [1]
    assert lease.depth == 0


@pytest.mark.parametrize(
    "lock_failure",
    [Timeout("provenance lease"), OSError("lock unavailable"), NotImplementedError("lock unsupported")],
)
def test_run_spec_wraps_writer_lease_acquisition_failure_before_plugin_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_failure: BaseException,
) -> None:
    decl, runtime = _synthetic_decl_and_runtime(tmp_path)
    lease = _TrackingLock(failure=lock_failure)
    calls: list[str] = []

    def _unexpected_run(self, ctx, inputs, cfg):
        del self, ctx, inputs, cfg
        calls.append("run")
        return {}

    monkeypatch.setattr(RecordStore, "provenance_lock", property(lambda _store: lease))
    monkeypatch.setattr(_SyntheticIngest, "run", _unexpected_run)

    with pytest.raises(ExecutionError, match="writer lease"):
        run_spec(
            decl,
            include_pipeline=True,
            include_plots=False,
            include_exports=False,
            log_level="ERROR",
            console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
            runtime=runtime,
        )

    assert calls == []


def test_run_spec_classifies_writer_lease_release_failure_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decl, runtime = _synthetic_decl_and_runtime(tmp_path)
    lease = _TrackingLock(release_failure=OSError("synthetic release failure"))
    monkeypatch.setattr(RecordStore, "provenance_lock", property(lambda _store: lease))

    with pytest.raises(ExecutionError, match="completed.*verify before continuing"):
        run_spec(
            decl,
            include_pipeline=True,
            include_plots=False,
            include_exports=False,
            log_level="ERROR",
            console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
            runtime=runtime,
        )

    assert (tmp_path / "outputs" / "manifests" / "records.json").is_file()
    assert lease.depth == 0


def test_run_spec_preserves_operation_error_when_writer_lease_release_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decl, runtime = _synthetic_decl_and_runtime(tmp_path)
    lease = _TrackingLock(release_failure=OSError("synthetic release failure"))

    def _fail_plugin(self, ctx, inputs, cfg):
        del self, ctx, inputs, cfg
        raise ValueError("plugin failure remains primary")

    monkeypatch.setattr(RecordStore, "provenance_lock", property(lambda _store: lease))
    monkeypatch.setattr(_SyntheticIngest, "run", _fail_plugin)

    with pytest.raises(ExecutionError, match="plugin failure remains primary") as raised:
        run_spec(
            decl,
            include_pipeline=True,
            include_plots=False,
            include_exports=False,
            log_level="ERROR",
            console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
            runtime=runtime,
        )

    assert any("also could not release the experiment writer lease" in note for note in raised.value.__notes__)
    assert lease.depth == 0


def test_run_spec_preserves_committed_records_when_success_result_cannot_be_confirmed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    decl, runtime = _synthetic_decl_and_runtime(tmp_path)

    def _fail_result(*_args, **_kwargs) -> None:
        raise ExecutionError("ledger unavailable")

    monkeypatch.setattr(InvocationLedger, "append_result", _fail_result)

    with pytest.raises(InvocationFinalizationError, match="records were committed") as raised:
        run_spec(
            decl,
            include_pipeline=True,
            include_plots=False,
            include_exports=False,
            log_level="ERROR",
            console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
            runtime=runtime,
        )

    store = runtime.record_store(outputs, experiment_root=tmp_path, create=False)
    assert store.latest_dataframe("ingest/df") is not None
    ledger_path = store.invocation_ledger_path()
    assert [event["event"] for event in _events(ledger_path)] == ["attempt"]
    assert raised.value.invocation_id
    assert raised.value.produced_record_revisions[0]["record_id"] == "ingest/df"


def test_run_spec_reset_records_replaces_retired_catalog_before_full_rerun(tmp_path: Path) -> None:
    decl, runtime = _synthetic_decl_and_runtime(tmp_path)
    records_path = tmp_path / "outputs" / "manifests" / "records.json"
    records_path.parent.mkdir(parents=True)
    retired_record = {"schema_version": 4, "record_id": "legacy/df"}
    records_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "latest": {"legacy/df": retired_record},
                "history": {"legacy/df": [retired_record]},
            }
        ),
        encoding="utf-8",
    )

    result = run_spec(
        decl,
        reset_records=True,
        include_pipeline=True,
        include_plots=False,
        include_exports=False,
        log_level="ERROR",
        console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
        runtime=runtime,
    )

    catalog = json.loads(records_path.read_text(encoding="utf-8"))
    assert result.status == "succeeded"
    assert set(catalog["latest"]) == {"ingest/df"}
    assert catalog["latest"]["ingest/df"]["schema_version"] == 5


def test_run_spec_reset_records_preserves_ledger_when_catalog_reset_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decl, runtime = _synthetic_decl_and_runtime(tmp_path)
    run_spec(
        decl,
        include_pipeline=True,
        include_plots=False,
        include_exports=False,
        log_level="ERROR",
        console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
        runtime=runtime,
    )
    store = runtime.record_store(tmp_path / "outputs", experiment_root=tmp_path, create=False)
    ledger_path = store.invocation_ledger_path()
    prior_ledger = ledger_path.read_bytes()

    def _fail_catalog_reset(_store: RecordStore) -> None:
        raise RecordError("catalog reset failed")

    monkeypatch.setattr(RecordStore, "reset_catalog", _fail_catalog_reset)

    with pytest.raises(RecordError, match="catalog reset failed"):
        run_spec(
            decl,
            reset_records=True,
            include_pipeline=True,
            include_plots=False,
            include_exports=False,
            log_level="ERROR",
            console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
            runtime=runtime,
        )

    assert ledger_path.read_bytes() == prior_ledger


def test_run_spec_reset_records_starts_a_new_invocation_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decl, runtime = _synthetic_decl_and_runtime(tmp_path)
    run_spec(
        decl,
        include_pipeline=True,
        include_plots=False,
        include_exports=False,
        log_level="ERROR",
        console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
        runtime=runtime,
    )
    store = runtime.record_store(tmp_path / "outputs", experiment_root=tmp_path, create=False)
    prior_epoch_id = store.provenance_epoch_id()
    prior_ledger_path = store.invocation_ledger_path()
    prior_ledger = prior_ledger_path.read_bytes()

    def _changed_run(self, ctx, inputs, cfg):
        del self, ctx, inputs, cfg
        return {"df": pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [2.0]})}

    monkeypatch.setattr(_SyntheticIngest, "run", _changed_run)
    result = run_spec(
        decl,
        reset_records=True,
        include_pipeline=True,
        include_plots=False,
        include_exports=False,
        log_level="ERROR",
        console=Console(quiet=True, theme=Theme({"accent": "cyan", "path": "magenta"})),
        runtime=runtime,
    )

    current_epoch_id = store.provenance_epoch_id()
    current_ledger_path = store.invocation_ledger_path()
    events = _events(current_ledger_path)
    assert current_epoch_id != prior_epoch_id
    assert current_ledger_path != prior_ledger_path
    assert prior_ledger_path.read_bytes() == prior_ledger
    assert [event["event"] for event in events] == ["attempt", "result"]
    assert {event["invocation_id"] for event in events} == {result.invocation_id}
    assert {event["provenance_epoch_id"] for event in events} == {current_epoch_id}

    record = store.latest_record("ingest/df")
    assert record is not None
    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest=record.config_digest,
    )
    assert report["status"] == "ok"
    assert report["summary"]["invocations_checked"] == 1
    assert report["summary"]["invocation_failures"] == 0


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
