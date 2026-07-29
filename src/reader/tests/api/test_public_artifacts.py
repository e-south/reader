from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from reader.api import ArtifactBundleResult, ArtifactSpec, Experiment, open_experiment, publish_artifact_bundle
from reader.errors import ExecutionError, InvocationFinalizationError, RecordError
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl, build_workbench_decl
from reader.workbench.engine.invocations import InvocationLedger
from reader.workbench.records import FileBundleRecord, RecordStore, verify_record_store


@dataclass(frozen=True)
class _Fixture:
    experiment: Experiment
    runtime: ReaderRuntime
    declaration: WorkbenchDecl
    store: RecordStore

    @property
    def outputs_dir(self) -> Path:
        return self.declaration.experiment_semantics.layout.outputs_dir

    @property
    def ledger_path(self) -> Path:
        return self.outputs_dir / "manifests" / "invocations.jsonl"


def _fixture(tmp_path: Path, *, exports_subdir: str = "exports") -> _Fixture:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema: reader/v8",
                "experiment:",
                "  id: cytometry_sink_test",
                "protocol:",
                "  id: cytometry/flow_panel",
                "resources:",
                "  metadata:",
                "    kind: file",
                "    path: ./inputs/metadata.csv",
                "paths:",
                "  outputs: ./generated",
                f"  exports: {exports_subdir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = builtin_runtime()
    spec = ReaderSpec.load(config_path)
    declaration = build_workbench_decl(spec, source_path=config_path, protocols=runtime.protocols)
    layout = declaration.experiment_semantics.layout
    store = runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=declaration.experiment.root,
    )
    return _Fixture(
        experiment=open_experiment(config_path, runtime=runtime),
        runtime=runtime,
        declaration=declaration,
        store=store,
    )


def _upstream_record(fixture: _Fixture) -> None:
    fixture.store.persist_dataframe(
        producer_id="merge_metadata",
        producer_plugin="transform/sample_metadata",
        out_name="df",
        record_id="merged/df",
        df=pd.DataFrame(
            {
                "position": ["sample-1"],
                "time": [0.0],
                "channel": ["RFP"],
                "value": [1.0],
            }
        ),
        contract_id="tidy.v1",
        inputs=[],
        config_digest=fixture.declaration.config_digest,
    )


def _artifact(relative_path: str, payload: str) -> ArtifactSpec:
    return ArtifactSpec(
        relative_path=relative_path,
        description=f"Artifact {relative_path}.",
        writer=lambda path: path.write_text(payload, encoding="utf-8"),
    )


def _publish(
    fixture: _Fixture,
    *,
    artifacts: tuple[ArtifactSpec, ...],
    upstream_records: dict[str, str] | None = None,
) -> ArtifactBundleResult:
    return publish_artifact_bundle(
        fixture.experiment,
        record_id="notebook:cytometry_eda",
        producer_id="cytometry_eda",
        template="notebook/cytometry",
        upstream_records=upstream_records or {"events": "merged/df"},
        producer_config={"threshold": 1.5, "gate": "configured"},
        description="Interactive cytometry plot, statistics, and gate definition.",
        artifacts=artifacts,
    )


def _ledger_events(fixture: _Fixture) -> list[dict[str, object]]:
    return [json.loads(line) for line in fixture.ledger_path.read_text(encoding="utf-8").splitlines()]


def test_publish_artifact_bundle_writes_verified_record_and_attempt_result_ledger(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, exports_subdir="review_exports")
    _upstream_record(fixture)

    result = _publish(
        fixture,
        artifacts=(
            _artifact("cytometry_eda.pdf", "pdf"),
            _artifact("cytometry_stats.csv", "sample,value\n1,2\n"),
            _artifact("cytometry_gates.json", json.dumps({"gate": 1.5})),
        ),
    )

    assert isinstance(result, ArtifactBundleResult)
    assert result.experiment == fixture.experiment.identity
    assert result.record.record_id == "notebook:cytometry_eda"
    assert result.record.revision == 1
    assert result.record.revision_digest.startswith("sha256:")
    paths = tuple(Path(path) for path in result.paths)
    assert {path.name for path in paths} == {
        "cytometry_eda.pdf",
        "cytometry_stats.csv",
        "cytometry_gates.json",
    }
    assert {path.parent.name for path in paths} == {"cytometry_eda"}
    assert all(path.is_file() for path in paths)
    assert all(path.is_relative_to(fixture.outputs_dir / "review_exports") for path in paths)

    restored = fixture.store.read_record("notebook:cytometry_eda")
    assert isinstance(restored, FileBundleRecord)
    assert restored.producer.kind == "notebook"
    assert restored.producer.template == "notebook/cytometry"
    assert restored.files == paths
    assert len(restored.file_evidence) == 3
    assert len(restored.inputs) == 1
    assert restored.inputs[0].ref.record_id == "merged/df"
    verification = verify_record_store(
        fixture.store,
        experiment_root=fixture.declaration.experiment.root,
        expected_config_digest=fixture.declaration.config_digest,
    )
    assert verification["status"] == "ok"

    events = _ledger_events(fixture)
    assert [event["event"] for event in events] == ["attempt", "result"]
    assert {event["invocation_id"] for event in events} == {result.invocation_id}
    assert events[0]["selected_step_ids"] == {
        "pipeline": [],
        "plots": [],
        "exports": ["cytometry_eda"],
    }
    assert events[1]["status"] == "succeeded"
    assert events[1]["produced_record_revisions"] == [result.record.to_dict()]
    assert result.ledger_path == str(fixture.ledger_path)


@pytest.mark.parametrize("relative_path", ["/tmp/escape.pdf", "../escape.pdf", "."])
def test_publish_artifact_bundle_rejects_unconfined_paths_before_writers(
    tmp_path: Path,
    relative_path: str,
) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)
    calls: list[Path] = []

    with pytest.raises(RecordError, match="relative|confined|file"):
        _publish(
            fixture,
            artifacts=(
                ArtifactSpec(
                    relative_path=relative_path,
                    description="Escaping artifact.",
                    writer=lambda path: calls.append(path),
                ),
            ),
        )

    assert calls == []
    assert not (fixture.outputs_dir / "exports" / "cytometry_eda").exists()
    assert not fixture.ledger_path.exists()


def test_publish_artifact_bundle_rejects_symlinked_staging_parent_before_writers(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)
    outside = tmp_path / "outside"
    outside.mkdir()
    staging = fixture.outputs_dir / ".staging"
    staging.symlink_to(outside, target_is_directory=True)
    calls: list[Path] = []

    with pytest.raises(RecordError, match="staging directory must stay within"):
        _publish(
            fixture,
            artifacts=(ArtifactSpec("cytometry_eda.pdf", "Plot artifact.", calls.append),),
        )

    assert calls == []
    assert list(outside.iterdir()) == []
    assert staging.is_symlink()
    _assert_failed_ledger(fixture, failure_type="RecordError")


def test_publish_artifact_bundle_rechecks_exports_confinement_after_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)
    outside = tmp_path / "outside"
    outside.mkdir()
    original_replace = Path.replace
    escaped_promotions: list[Path] = []

    def _track_replace(source: Path, target: Path) -> Path:
        if source.parent == fixture.outputs_dir / ".staging":
            resolved_target = target.resolve(strict=False)
            if not resolved_target.is_relative_to(fixture.outputs_dir.resolve(strict=True)):
                escaped_promotions.append(resolved_target)
        return original_replace(source, target)

    def _replace_exports_with_symlink(path: Path) -> None:
        path.write_text("pdf", encoding="utf-8")
        fixture.store.exports_dir.rmdir()
        fixture.store.exports_dir.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(Path, "replace", _track_replace)

    with pytest.raises(RecordError, match="exports.*within|symlink"):
        _publish(
            fixture,
            artifacts=(
                ArtifactSpec(
                    "cytometry_eda.pdf",
                    "Plot artifact.",
                    _replace_exports_with_symlink,
                ),
            ),
        )

    assert escaped_promotions == []
    assert list(outside.iterdir()) == []
    assert fixture.store.latest_record("notebook:cytometry_eda") is None
    _assert_clean_staging(fixture)
    _assert_failed_ledger(fixture, failure_type="RecordError")


def test_publish_artifact_bundle_rejects_symlink_escape_and_cleans_staging(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)
    outside = tmp_path / "outside"
    outside.mkdir()

    def _escape(path: Path) -> None:
        path.symlink_to(outside / "escaped.pdf")

    with pytest.raises(RecordError, match="confined|regular file"):
        _publish(fixture, artifacts=(ArtifactSpec("cytometry_eda.pdf", "Plot artifact.", _escape),))

    assert not (outside / "escaped.pdf").exists()
    _assert_no_partial_bundle(fixture)
    _assert_failed_ledger(fixture, failure_type="RecordError")


def test_publish_artifact_bundle_rejects_internal_symlink_alias(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)

    def _alias(path: Path) -> None:
        target = path.with_name("target.pdf")
        target.write_text("pdf", encoding="utf-8")
        path.symlink_to(target.name)

    with pytest.raises(RecordError, match="regular file"):
        _publish(fixture, artifacts=(ArtifactSpec("cytometry_eda.pdf", "Plot artifact.", _alias),))

    _assert_no_partial_bundle(fixture)
    _assert_failed_ledger(fixture, failure_type="RecordError")


def test_publish_artifact_bundle_rejects_untracked_and_empty_writer_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)

    def _write_extra(path: Path) -> None:
        path.write_text("pdf", encoding="utf-8")
        path.with_name("untracked.txt").write_text("untracked", encoding="utf-8")

    with pytest.raises(RecordError, match="exactly the declared non-empty files"):
        _publish(fixture, artifacts=(ArtifactSpec("cytometry_eda.pdf", "Plot artifact.", _write_extra),))
    with pytest.raises(RecordError, match="non-empty|exactly the declared"):
        _publish(fixture, artifacts=(_artifact("cytometry_eda.pdf", ""),))

    _assert_no_partial_bundle(fixture)
    events = _ledger_events(fixture)
    assert [event["event"] for event in events] == ["attempt", "result", "attempt", "result"]
    assert all(event["status"] == "failed" for event in events if event["event"] == "result")


def test_publish_artifact_bundle_writer_failure_rolls_back_and_records_failed_result(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)

    def _fail(path: Path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        _publish(
            fixture,
            artifacts=(
                _artifact("cytometry_eda.pdf", "pdf"),
                ArtifactSpec("cytometry_stats.csv", "Statistics artifact.", _fail),
                _artifact("cytometry_gates.json", "{}"),
            ),
        )

    assert fixture.store.latest_record("notebook:cytometry_eda") is None
    _assert_no_partial_bundle(fixture)
    _assert_failed_ledger(fixture, failure_type="RuntimeError")


@pytest.mark.parametrize("failure", [RecordError("catalog failed"), KeyboardInterrupt()])
def test_publish_artifact_bundle_catalog_failure_rolls_back_promoted_files_and_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)

    def _fail_catalog(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(RecordStore, "append_notebook_file_bundle", _fail_catalog)

    with pytest.raises(type(failure), match=str(failure) or None):
        _publish(fixture, artifacts=(_artifact("cytometry_eda.pdf", "pdf"),))

    assert fixture.store.latest_record("notebook:cytometry_eda") is None
    _assert_no_partial_bundle(fixture)
    _assert_failed_ledger(fixture, failure_type=type(failure).__name__)


def test_publish_artifact_bundle_collision_preserves_winning_revision(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)
    winning_dir = fixture.outputs_dir / "exports" / "cytometry_eda"
    winning_marker = winning_dir / "winner.pdf"
    original_replace = Path.replace

    def _collide(source: Path, target: Path):
        if source.parent == fixture.outputs_dir / ".staging":
            winning_dir.mkdir(parents=True)
            winning_marker.write_text("winner", encoding="utf-8")
            raise FileExistsError("another publisher won the revision")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", _collide)

    with pytest.raises(FileExistsError, match="another publisher won"):
        _publish(fixture, artifacts=(_artifact("cytometry_eda.pdf", "loser"),))

    assert winning_marker.read_text(encoding="utf-8") == "winner"
    _assert_clean_staging(fixture)
    _assert_failed_ledger(fixture, failure_type="FileExistsError")


def test_publish_artifact_bundle_requires_existing_upstream_record_before_attempt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)

    with pytest.raises(RecordError, match="Input record 'missing/df' is missing"):
        _publish(
            fixture,
            artifacts=(_artifact("cytometry_eda.pdf", "pdf"),),
            upstream_records={"events": "missing/df"},
        )

    assert not (fixture.outputs_dir / "exports" / "cytometry_eda").exists()
    assert not fixture.ledger_path.exists()


def test_publish_artifact_bundle_rejects_dataframe_record_id_before_attempt_or_writer(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)
    fixture.store.persist_dataframe(
        producer_id="summary",
        producer_plugin="transform/sample_metadata",
        out_name="df",
        record_id="summary/df",
        df=pd.DataFrame(
            {
                "position": ["sample-1"],
                "time": [0.0],
                "channel": ["RFP"],
                "value": [1.0],
            }
        ),
        contract_id="tidy.v1",
        inputs=[],
        config_digest=fixture.declaration.config_digest,
    )
    calls: list[Path] = []

    with pytest.raises(RecordError, match="already used by a dataframe record"):
        publish_artifact_bundle(
            fixture.experiment,
            record_id="summary/df",
            producer_id="cytometry_eda",
            template="notebook/cytometry",
            upstream_records={"events": "merged/df"},
            producer_config={"threshold": 1.5},
            description="Interactive cytometry plot.",
            artifacts=(ArtifactSpec("cytometry_eda.pdf", "Plot artifact.", calls.append),),
        )

    assert calls == []
    assert fixture.store.record_history("summary/df") == (fixture.store.read_record("summary/df"),)
    assert not (fixture.outputs_dir / "exports" / "cytometry_eda").exists()
    assert not fixture.ledger_path.exists()


def test_publish_artifact_bundle_rejects_self_reference_before_attempt_or_writer(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)
    calls: list[Path] = []

    with pytest.raises(RecordError, match="must not reference itself"):
        _publish(
            fixture,
            artifacts=(ArtifactSpec("cytometry_eda.pdf", "Plot artifact.", calls.append),),
            upstream_records={"events": "notebook:cytometry_eda"},
        )

    assert calls == []
    assert fixture.store.latest_record("notebook:cytometry_eda") is None
    assert not (fixture.outputs_dir / "exports" / "cytometry_eda").exists()
    assert not fixture.ledger_path.exists()


def test_changed_artifact_bundle_uses_immutable_revision_directory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)

    first = _publish(fixture, artifacts=(_artifact("cytometry_eda.pdf", "first"),))
    second = _publish(fixture, artifacts=(_artifact("cytometry_eda.pdf", "second"),))
    first_path, second_path = Path(first.paths[0]), Path(second.paths[0])

    assert first_path.parent.name == "cytometry_eda"
    assert second_path.parent.name == "cytometry_eda__r2"
    assert first_path.read_text(encoding="utf-8") == "first"
    assert second_path.read_text(encoding="utf-8") == "second"
    assert first.record.revision == 1
    assert second.record.revision == 2


def test_success_result_failure_preserves_committed_artifact_without_false_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)

    def _fail_result(*_args, **_kwargs) -> None:
        raise ExecutionError("ledger unavailable")

    monkeypatch.setattr(InvocationLedger, "append_result", _fail_result)

    with pytest.raises(InvocationFinalizationError, match="records were committed") as raised:
        _publish(fixture, artifacts=(_artifact("cytometry_eda.pdf", "pdf"),))

    record = fixture.store.read_record("notebook:cytometry_eda")
    assert isinstance(record, FileBundleRecord)
    assert all(path.is_file() for path in record.files)
    assert [event["event"] for event in _ledger_events(fixture)] == ["attempt"]
    assert raised.value.invocation_id
    assert raised.value.produced_record_revisions[0]["record_id"] == record.record_id
    assert raised.value.produced_record_revisions[0]["revision"] == 1
    verification = verify_record_store(
        fixture.store,
        experiment_root=fixture.declaration.experiment.root,
        expected_config_digest=fixture.declaration.config_digest,
    )
    assert verification["status"] == "failed"
    assert verification["summary"]["invocation_failures"] == 1
    assert verification["issues"][0]["code"] == "invocation.finalization_unconfirmed"


def test_success_result_acknowledgement_failure_does_not_append_a_failed_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)
    original = InvocationLedger.append_result

    def _append_then_fail(self, attempt, **kwargs) -> None:
        original(self, attempt, **kwargs)
        raise ExecutionError("result acknowledgement failed")

    monkeypatch.setattr(InvocationLedger, "append_result", _append_then_fail)

    with pytest.raises(InvocationFinalizationError, match="could not confirm"):
        _publish(fixture, artifacts=(_artifact("cytometry_eda.pdf", "pdf"),))

    record = fixture.store.read_record("notebook:cytometry_eda")
    assert isinstance(record, FileBundleRecord)
    assert all(path.is_file() for path in record.files)
    events = _ledger_events(fixture)
    assert [event["event"] for event in events] == ["attempt", "result"]
    assert events[1]["status"] == "succeeded"


def test_precommit_failure_remains_primary_when_failure_result_cannot_be_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _upstream_record(fixture)

    def _fail_writer(path: Path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("writer failed")

    def _fail_result(*_args, **_kwargs) -> None:
        raise ExecutionError("ledger unavailable")

    monkeypatch.setattr(InvocationLedger, "append_result", _fail_result)

    with pytest.raises(RuntimeError, match="writer failed") as raised:
        _publish(
            fixture,
            artifacts=(ArtifactSpec("cytometry_eda.pdf", "Plot artifact.", _fail_writer),),
        )

    assert any("could not persist the failed invocation result" in note for note in raised.value.__notes__)
    assert fixture.store.latest_record("notebook:cytometry_eda") is None
    _assert_no_partial_bundle(fixture)
    assert [event["event"] for event in _ledger_events(fixture)] == ["attempt"]


def _assert_no_partial_bundle(fixture: _Fixture) -> None:
    assert not (fixture.outputs_dir / "exports" / "cytometry_eda").exists()
    _assert_clean_staging(fixture)


def _assert_clean_staging(fixture: _Fixture) -> None:
    staging = fixture.outputs_dir / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def _assert_failed_ledger(fixture: _Fixture, *, failure_type: str) -> None:
    events = _ledger_events(fixture)
    assert [event["event"] for event in events] == ["attempt", "result"]
    assert events[0]["invocation_id"] == events[1]["invocation_id"]
    assert events[1]["status"] == "failed"
    assert events[1]["exit_status"] == 1
    assert events[1]["produced_record_revisions"] == []
    assert events[1]["failure"]["type"] == failure_type
