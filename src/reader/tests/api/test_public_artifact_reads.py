from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

import reader.api.artifacts as artifact_api
from reader.api import ArtifactFileResult, Experiment, open_experiment, read_artifact
from reader.errors import RecordError
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.tests.support.configs import base_reader_config, write_config
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl, build_workbench_decl
from reader.workbench.records import PathDescription, record_revision_digest


@dataclass(frozen=True)
class _Fixture:
    experiment: Experiment
    runtime: ReaderRuntime
    declaration: WorkbenchDecl


def _fixture(tmp_path: Path) -> _Fixture:
    config_path = write_config(tmp_path / "config.yaml", base_reader_config(experiment_id="example"))
    runtime = builtin_runtime()
    spec = ReaderSpec.load(config_path)
    declaration = build_workbench_decl(spec, source_path=config_path, protocols=runtime.protocols)
    return _Fixture(
        experiment=open_experiment(config_path, runtime=runtime),
        runtime=runtime,
        declaration=declaration,
    )


def _store(fixture: _Fixture):
    layout = fixture.declaration.experiment_semantics.layout
    return fixture.runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=fixture.declaration.experiment.root,
    )


def _append_plot(store, fixture: _Fixture, *, revision_name: str, content: bytes):
    path = store.plots_dir / revision_name / "summary.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    record = store.append_file_bundle(
        producer_kind="plot",
        producer_id="summary",
        producer_plugin="plot/time_series",
        record_id="plot:summary",
        inputs=[],
        config_digest=fixture.declaration.config_digest,
        files=[path],
        description="Summary plot.",
        path_descriptions=(PathDescription(path=path, description="Summary plot."),),
    )
    return record, path


def test_read_artifact_returns_verified_bytes_for_one_exact_revision(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    store = _store(fixture)
    first, first_path = _append_plot(store, fixture, revision_name="r1", content=b"%PDF first")
    first_digest = record_revision_digest(first, outputs_dir=store.root)
    _append_plot(store, fixture, revision_name="r2", content=b"%PDF second")

    result = read_artifact(
        fixture.experiment,
        first.record_id,
        revision=1,
        revision_digest=first_digest,
        path=first_path.relative_to(store.root).as_posix(),
    )

    assert isinstance(result, ArtifactFileResult)
    assert result.record.revision == 1
    assert result.record.revision_digest == first_digest
    assert result.relative_path == first_path.relative_to(store.root).as_posix()
    assert result.content == b"%PDF first"
    assert result.size_bytes == len(result.content)
    assert result.content_digest.startswith("sha256:")
    assert result.media_type == "application/pdf"
    assert "content" not in result.to_dict()


def test_read_artifact_rejects_revision_drift_and_tampered_content(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    store = _store(fixture)
    first, first_path = _append_plot(store, fixture, revision_name="r1", content=b"image-v1")
    second, _ = _append_plot(store, fixture, revision_name="r2", content=b"image-v2")
    first_digest = record_revision_digest(first, outputs_dir=store.root)
    second_digest = record_revision_digest(second, outputs_dir=store.root)
    relative_path = first_path.relative_to(store.root).as_posix()

    with pytest.raises(RecordError, match="revision digest mismatch.*refresh"):
        read_artifact(
            fixture.experiment,
            first.record_id,
            revision=1,
            revision_digest=second_digest,
            path=relative_path,
        )

    first_path.write_bytes(b"tampered")
    with pytest.raises(RecordError, match="content digest mismatch"):
        read_artifact(
            fixture.experiment,
            first.record_id,
            revision=1,
            revision_digest=first_digest,
            path=relative_path,
        )


def test_read_artifact_rejects_missing_cataloged_file(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    store = _store(fixture)
    record, path = _append_plot(store, fixture, revision_name="r1", content=b"plot")
    revision_digest = record_revision_digest(record, outputs_dir=store.root)
    relative_path = path.relative_to(store.root).as_posix()
    path.unlink()

    with pytest.raises(RecordError, match="artifact is missing"):
        read_artifact(
            fixture.experiment,
            record.record_id,
            revision=1,
            revision_digest=revision_digest,
            path=relative_path,
        )


@pytest.mark.parametrize("path", ["/private/result.pdf", "../result.pdf", "plots/../result.pdf", ""])
def test_read_artifact_rejects_unconfined_or_missing_paths(tmp_path: Path, path: str) -> None:
    fixture = _fixture(tmp_path)
    store = _store(fixture)
    record, _ = _append_plot(store, fixture, revision_name="r1", content=b"plot")

    with pytest.raises(RecordError, match="path must"):
        read_artifact(
            fixture.experiment,
            record.record_id,
            revision=1,
            revision_digest=record_revision_digest(record, outputs_dir=store.root),
            path=path,
        )


def test_read_artifact_rejects_confined_file_outside_selected_bundle(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    store = _store(fixture)
    record, _ = _append_plot(store, fixture, revision_name="r1", content=b"plot")
    outsider = store.plots_dir / "other.pdf"
    outsider.write_bytes(b"other")

    with pytest.raises(RecordError, match="is not part of record"):
        read_artifact(
            fixture.experiment,
            record.record_id,
            revision=1,
            revision_digest=record_revision_digest(record, outputs_dir=store.root),
            path=outsider.relative_to(store.root).as_posix(),
        )


def test_read_artifact_rejects_dataframe_records(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    store = _store(fixture)
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest=fixture.declaration.config_digest,
    )

    with pytest.raises(RecordError, match="is not a file bundle"):
        read_artifact(
            fixture.experiment,
            record.record_id,
            revision=1,
            revision_digest=record_revision_digest(record, outputs_dir=store.root),
            path="plots/result.pdf",
        )


@pytest.mark.parametrize(
    ("revision", "revision_digest", "message"),
    [
        (None, None, "requires both revision and revision_digest"),
        (None, "sha256:" + "a" * 64, "requires both revision and revision_digest"),
        (1, None, "requires both revision and revision_digest"),
        (0, "sha256:" + "a" * 64, "positive integer"),
        (True, "sha256:" + "a" * 64, "positive integer"),
        (1, "sha256:" + "z" * 64, "sha256 digest"),
        (2, "sha256:" + "a" * 64, "revision 2 is unavailable"),
    ],
)
def test_read_artifact_rejects_invalid_revision_identity(
    tmp_path: Path,
    revision,
    revision_digest,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    store = _store(fixture)
    record, path = _append_plot(store, fixture, revision_name="r1", content=b"plot")

    with pytest.raises(RecordError, match=message):
        read_artifact(
            fixture.experiment,
            record.record_id,
            revision=revision,
            revision_digest=revision_digest,
            path=path.relative_to(store.root).as_posix(),
        )


def test_read_artifact_rechecks_the_exact_returned_bytes_after_integrity_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    store = _store(fixture)
    record, path = _append_plot(store, fixture, revision_name="r1", content=b"plot")
    original_verify = artifact_api.verify_record_artifact_integrity

    def _verify_then_mutate(*args, **kwargs) -> None:
        original_verify(*args, **kwargs)
        path.write_bytes(b"changed after verification")

    monkeypatch.setattr(artifact_api, "verify_record_artifact_integrity", _verify_then_mutate)

    with pytest.raises(RecordError, match="changed while it was being read"):
        read_artifact(
            fixture.experiment,
            record.record_id,
            revision=1,
            revision_digest=record_revision_digest(record, outputs_dir=store.root),
            path=path.relative_to(store.root).as_posix(),
        )
