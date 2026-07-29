from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from reader.errors import RecordError
from reader.runtime import builtin_runtime
from reader.workbench.notebooks import (
    NotebookArtifactSpec,
    publish_notebook_artifact_bundle,
)
from reader.workbench.notebooks.context import load_notebook_workbench_context
from reader.workbench.records import FileBundleRecord, verify_record_store
from reader.workbench.records.store import RecordStore


def _context(tmp_path: Path, *, exports_subdir: str = "exports"):
    config = tmp_path / "config.yaml"
    config.write_text(
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
    return load_notebook_workbench_context(config)


def _upstream_record(context) -> None:
    runtime = builtin_runtime()
    layout = context.decl.experiment_semantics.layout
    store = runtime.record_store(
        context.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=context.experiment_root,
    )
    store.persist_dataframe(
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
        config_digest=context.decl.config_digest,
    )


def _artifact(relative_path: str, payload: str) -> NotebookArtifactSpec:
    return NotebookArtifactSpec(
        relative_path=relative_path,
        description=f"Artifact {relative_path}.",
        writer=lambda path: path.write_text(payload, encoding="utf-8"),
    )


def _publish(context, *, artifacts: tuple[NotebookArtifactSpec, ...]):
    return publish_notebook_artifact_bundle(
        context,
        record_id="notebook:cytometry_eda",
        producer_id="cytometry_eda",
        template="notebook/cytometry",
        upstream_records={"events": "merged/df"},
        producer_config={"threshold": 1.5, "gate": "configured"},
        description="Interactive cytometry plot, statistics, and gate definition.",
        artifacts=artifacts,
    )


def test_publish_notebook_bundle_writes_three_files_and_one_typed_record(tmp_path: Path) -> None:
    context = _context(tmp_path, exports_subdir="review_exports")
    _upstream_record(context)

    record = _publish(
        context,
        artifacts=(
            _artifact("cytometry_eda.pdf", "pdf"),
            _artifact("cytometry_stats.csv", "sample,value\n1,2\n"),
            _artifact("cytometry_gates.json", json.dumps({"gate": 1.5})),
        ),
    )

    assert isinstance(record, FileBundleRecord)
    assert record.schema_version == 5
    assert record.producer.kind == "notebook"
    assert record.producer.id == "cytometry_eda"
    assert record.producer.template == "notebook/cytometry"
    assert record.producer.plugin is None
    assert record.record_id == "notebook:cytometry_eda"
    assert {path.name for path in record.files} == {
        "cytometry_eda.pdf",
        "cytometry_stats.csv",
        "cytometry_gates.json",
    }
    assert {path.parent.name for path in record.files} == {"cytometry_eda"}
    assert all(path.is_file() for path in record.files)
    assert all(path.is_relative_to(context.outputs_dir / "review_exports") for path in record.files)
    assert len(record.file_evidence) == 3
    assert len(record.inputs) == 1
    assert record.inputs[0].ref.record_id == "merged/df"

    store = builtin_runtime().record_store(
        context.outputs_dir,
        experiment_root=context.experiment_root,
        create=False,
    )
    restored = store.read_record("notebook:cytometry_eda")
    assert restored == record
    assert len(store.record_history("notebook:cytometry_eda")) == 1
    verification = verify_record_store(
        store,
        experiment_root=context.experiment_root,
        expected_config_digest=context.decl.config_digest,
    )
    assert verification["status"] == "ok"


@pytest.mark.parametrize("relative_path", ["/tmp/escape.pdf", "../escape.pdf", "."])
def test_publish_notebook_bundle_rejects_unconfined_paths_before_writers(tmp_path: Path, relative_path: str) -> None:
    context = _context(tmp_path)
    _upstream_record(context)
    calls: list[Path] = []

    with pytest.raises(RecordError, match="relative|confined|file"):
        artifact = NotebookArtifactSpec(
            relative_path=relative_path,
            description="Escaping artifact.",
            writer=lambda path: calls.append(path),
        )
        _publish(context, artifacts=(artifact,))

    assert calls == []
    assert not (context.outputs_dir / "exports" / "cytometry_eda").exists()


def test_publish_notebook_bundle_rejects_symlinked_staging_parent_before_writers(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _upstream_record(context)
    outside = tmp_path / "outside"
    outside.mkdir()
    staging = context.outputs_dir / ".staging"
    staging.symlink_to(outside, target_is_directory=True)
    calls: list[Path] = []

    with pytest.raises(RecordError, match="staging directory must stay within"):
        _publish(
            context,
            artifacts=(
                NotebookArtifactSpec(
                    relative_path="cytometry_eda.pdf",
                    description="Plot artifact.",
                    writer=lambda path: calls.append(path),
                ),
            ),
        )

    assert calls == []
    assert list(outside.iterdir()) == []
    assert staging.is_symlink()


def test_publish_notebook_bundle_rejects_symlink_escape_and_cleans_staging(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _upstream_record(context)
    outside = tmp_path / "outside"
    outside.mkdir()

    def _escape(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(outside / "escaped.pdf")

    with pytest.raises(RecordError, match="confined|regular file"):
        _publish(
            context,
            artifacts=(NotebookArtifactSpec("cytometry_eda.pdf", "Plot artifact.", _escape),),
        )

    assert not (outside / "escaped.pdf").exists()
    assert not (context.outputs_dir / "exports" / "cytometry_eda").exists()
    staging = context.outputs_dir / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_publish_notebook_bundle_rejects_internal_symlink_alias(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _upstream_record(context)

    def _alias(path: Path) -> None:
        target = path.with_name("target.pdf")
        target.write_text("pdf", encoding="utf-8")
        path.symlink_to(target.name)

    with pytest.raises(RecordError, match="regular file"):
        _publish(
            context,
            artifacts=(NotebookArtifactSpec("cytometry_eda.pdf", "Plot artifact.", _alias),),
        )

    assert not (context.outputs_dir / "exports" / "cytometry_eda").exists()


def test_publish_notebook_bundle_rejects_untracked_and_empty_writer_outputs(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _upstream_record(context)

    def _write_extra(path: Path) -> None:
        path.write_text("pdf", encoding="utf-8")
        path.with_name("untracked.txt").write_text("untracked", encoding="utf-8")

    with pytest.raises(RecordError, match="exactly the declared non-empty files"):
        _publish(
            context,
            artifacts=(NotebookArtifactSpec("cytometry_eda.pdf", "Plot artifact.", _write_extra),),
        )

    with pytest.raises(RecordError, match="exactly the declared non-empty files"):
        _publish(context, artifacts=(_artifact("cytometry_eda.pdf", ""),))

    assert not (context.outputs_dir / "exports" / "cytometry_eda").exists()


def test_publish_notebook_bundle_writer_failure_leaves_no_record_or_partial_bundle(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _upstream_record(context)

    def _fail(path: Path) -> None:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        _publish(
            context,
            artifacts=(
                _artifact("cytometry_eda.pdf", "pdf"),
                NotebookArtifactSpec("cytometry_stats.csv", "Statistics artifact.", _fail),
                _artifact("cytometry_gates.json", "{}"),
            ),
        )

    store = builtin_runtime().record_store(
        context.outputs_dir,
        experiment_root=context.experiment_root,
        create=False,
    )
    assert store.latest_record("notebook:cytometry_eda") is None
    assert not (context.outputs_dir / "exports" / "cytometry_eda").exists()
    staging = context.outputs_dir / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_publish_notebook_bundle_catalog_failure_rolls_back_committed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    _upstream_record(context)

    def _fail_catalog(*_args, **_kwargs):
        raise RecordError("catalog failed")

    monkeypatch.setattr(RecordStore, "append_notebook_file_bundle", _fail_catalog)

    with pytest.raises(RecordError, match="catalog failed"):
        _publish(context, artifacts=(_artifact("cytometry_eda.pdf", "pdf"),))

    assert not (context.outputs_dir / "exports" / "cytometry_eda").exists()
    staging = context.outputs_dir / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_publish_notebook_bundle_catalog_interrupt_rolls_back_promoted_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    _upstream_record(context)

    def _interrupt_catalog(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(RecordStore, "append_notebook_file_bundle", _interrupt_catalog)

    with pytest.raises(KeyboardInterrupt):
        _publish(context, artifacts=(_artifact("cytometry_eda.pdf", "pdf"),))

    assert not (context.outputs_dir / "exports" / "cytometry_eda").exists()
    staging = context.outputs_dir / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_publish_notebook_bundle_collision_never_deletes_winning_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    _upstream_record(context)
    winning_dir = context.outputs_dir / "exports" / "cytometry_eda"
    winning_marker = winning_dir / "winner.pdf"
    original_replace = Path.replace

    def _collide(source: Path, target: Path):
        if source.parent == context.outputs_dir / ".staging":
            winning_dir.mkdir(parents=True)
            winning_marker.write_text("winner", encoding="utf-8")
            raise FileExistsError("another publisher won the revision")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", _collide)

    with pytest.raises(FileExistsError, match="another publisher won"):
        _publish(context, artifacts=(_artifact("cytometry_eda.pdf", "loser"),))

    assert winning_marker.read_text(encoding="utf-8") == "winner"
    staging = context.outputs_dir / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_publish_notebook_bundle_requires_existing_upstream_record(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with pytest.raises(RecordError, match="Input record 'merged/df' is missing"):
        _publish(context, artifacts=(_artifact("cytometry_eda.pdf", "pdf"),))

    assert not (context.outputs_dir / "exports" / "cytometry_eda").exists()


def test_changed_notebook_bundle_uses_immutable_revision_directory(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _upstream_record(context)

    first = _publish(context, artifacts=(_artifact("cytometry_eda.pdf", "first"),))
    second = _publish(context, artifacts=(_artifact("cytometry_eda.pdf", "second"),))

    assert first.files[0].parent.name == "cytometry_eda"
    assert second.files[0].parent.name == "cytometry_eda__r2"
    assert first.files[0].read_text(encoding="utf-8") == "first"
    assert second.files[0].read_text(encoding="utf-8") == "second"
