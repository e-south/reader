from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reader.runtime import builtin_runtime
from reader.workbench.notebooks.components.deliverables import (
    NotebookDeliverables,
    build_notebook_deliverable_selector,
    collect_notebook_deliverables,
    render_notebook_deliverable_viewport,
)
from reader.workbench.records import PathDescription, record_revision_digest, record_to_dict


class _FakeUi:
    def dropdown(self, *, options, value, label, full_width):
        return type(
            "Dropdown",
            (),
            {"options": options, "value": options[value], "label": label, "full_width": full_width},
        )()

    def table(self, rows, *, page_size):
        return {"kind": "table", "rows": list(rows), "page_size": page_size}


class _FakeMarimo:
    def __init__(self) -> None:
        self.ui = _FakeUi()
        self.accordion_calls = []

    def md(self, text):
        return {"kind": "markdown", "text": text}

    def accordion(self, sections, *, multiple, lazy):
        self.accordion_calls.append({"sections": sections, "multiple": multiple, "lazy": lazy})
        return {"kind": "accordion", "sections": sections}

    def vstack(self, items):
        return {"kind": "vstack", "items": list(items)}

    def image(self, source, *, alt):
        return {"kind": "image", "source": source, "alt": alt}

    def pdf(self, source, *, width, height):
        return {"kind": "pdf", "source": source, "width": width, "height": height}


def test_collect_notebook_deliverables_summarizes_cataloged_records_plots_and_exports(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    plots = outputs / "plots"
    exports = outputs / "exports"
    plots.mkdir(parents=True)
    exports.mkdir(parents=True)
    plot_file = plots / "summary.pdf"
    second_plot_file = plots / "kinetics.pdf"
    export_file = exports / "summary.xlsx"
    plot_file.write_text("plot", encoding="utf-8")
    second_plot_file.write_text("plot", encoding="utf-8")
    export_file.write_text("export", encoding="utf-8")

    runtime = builtin_runtime()
    store = runtime.record_store(outputs, plots_subdir="plots", exports_subdir="exports")
    store.persist_dataframe(
        producer_id="summary",
        producer_plugin="transform/example",
        out_name="df",
        record_id="summary/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    store.append_file_bundle(
        producer_kind="plot",
        producer_id="plot_summary",
        producer_plugin="plot/time_series",
        record_id="plot:plot_summary",
        inputs=[],
        config_digest="sha256:test",
        files=[plot_file, second_plot_file],
        description="Render grouped time-series plots from tidy plate-reader traces.",
        path_descriptions=(
            PathDescription(path=plot_file, description="Endpoint summary by treatment."),
            PathDescription(path=second_plot_file, description="Reporter kinetics over assay time."),
        ),
    )
    store.append_file_bundle(
        producer_kind="export",
        producer_id="export_summary",
        producer_plugin="export/xlsx",
        record_id="export:export_summary",
        inputs=[],
        config_digest="sha256:test",
        files=[export_file],
        description="Write dataframe records to XLSX workbooks.",
    )
    records = store.iter_latest_records()
    entries = tuple(
        {
            **record_to_dict(record, outputs_dir=outputs),
            "revision": store.revision_counts([record.record_id])[record.record_id],
            "revision_digest": record_revision_digest(record, outputs_dir=outputs),
        }
        for record in records
    )
    deliverables = collect_notebook_deliverables(
        entries,
        outputs_dir=outputs,
        verification_status="ok",
        verification_issues=(),
        verification_records=tuple({"record_id": record.record_id, "status": "ok", "issues": []} for record in records),
    )

    assert {"Deliverable": "Dataframe records", "Count": 1} in deliverables.summary_rows
    assert {"Deliverable": "Plot files", "Count": 2} in deliverables.summary_rows
    assert {"Deliverable": "Export files", "Count": 1} in deliverables.summary_rows
    assert {"Deliverable": "Other artifact files", "Count": 0} in deliverables.summary_rows
    assert deliverables.record_rows[0]["Record ID"] == "summary/df"
    assert deliverables.record_rows[0]["Revision"] == 1
    assert str(deliverables.record_rows[0]["Revision digest"]).startswith("sha256:")
    assert deliverables.record_rows[0]["Verification"] == "ok"
    plot_rows = {row["Path"]: row for row in deliverables.plot_rows}
    assert plot_rows["plots/summary.pdf"]["Description"] == "Endpoint summary by treatment."
    assert plot_rows["plots/kinetics.pdf"]["Description"] == "Reporter kinetics over assay time."
    assert deliverables.export_rows[0]["Path"] == "exports/summary.xlsx"
    assert deliverables.export_rows[0]["Description"] == "Write dataframe records to XLSX workbooks."


def test_collect_notebook_deliverables_handles_empty_public_catalog(tmp_path: Path) -> None:
    deliverables = collect_notebook_deliverables(
        (),
        outputs_dir=tmp_path / "outputs",
        verification_status="failed",
        verification_issues=(
            {
                "code": "catalog.missing",
                "field": "outputs/manifests/records.json",
                "reason": f"Catalog is absent at {tmp_path / 'private' / 'records.json'}.",
                "remediation": "Run the experiment and verify again.",
                "retryable": False,
            },
        ),
        verification_records=(),
    )

    assert deliverables.record_rows == ()
    assert deliverables.plot_rows == ()
    assert deliverables.export_rows == ()
    assert deliverables.verification_status == "failed"
    assert deliverables.issue_rows == (
        {
            "Scope": "catalog",
            "Record ID": "",
            "Status": "failed",
            "Code": "catalog.missing",
            "Field": "outputs/manifests/records.json",
            "Action": "Run the experiment and verify again.",
            "Retryable": "no",
        },
    )
    assert str(tmp_path) not in str(deliverables.issue_rows)
    assert deliverables.artifact_rows == ()


def test_collect_notebook_deliverables_never_projects_absolute_or_traversing_paths(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    entries = (
        {
            "kind": "file_bundle",
            "record_id": "plot:outside",
            "producer": {"kind": "plot", "id": "outside", "plugin": "plot/time_series"},
            "description": "Unsafe catalog fixture.",
            "files": [str(tmp_path / "private-study" / "identity.png"), "../private-study/trace.png"],
            "revision": 1,
            "revision_digest": "sha256:" + "a" * 64,
        },
    )

    deliverables = collect_notebook_deliverables(
        entries,
        outputs_dir=outputs,
        verification_status="failed",
        verification_issues=(),
        verification_records=({"record_id": "plot:outside", "status": "failed", "issues": []},),
    )

    assert {row["Path"] for row in deliverables.plot_rows} == {"[outside outputs]"}
    assert str(tmp_path) not in str(deliverables)


def test_generic_notebook_components_surface_new_dataframe_contracts_and_pipeline_file_bundles(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    dataframe_path = outputs / "artifacts" / "novel" / "measurements.parquet"
    bundle_path = outputs / "artifacts" / "novel" / "readme.txt"
    dataframe_path.parent.mkdir(parents=True)
    dataframe_path.write_bytes(b"cataloged dataframe placeholder")
    bundle_path.write_text("domain-neutral artifact", encoding="utf-8")
    entries = (
        {
            "kind": "dataframe_artifact",
            "record_id": "novel/measurements",
            "contract_id": "novel.measurements.v1",
            "created_at": "2026-07-29T12:00:00Z",
            "producer": {"kind": "pipeline", "id": "novel", "plugin": "transform/novel"},
            "path": "artifacts/novel/measurements.parquet",
            "revision": 3,
            "revision_digest": "sha256:" + "a" * 64,
        },
        {
            "kind": "file_bundle",
            "record_id": "novel/supporting_files",
            "producer": {"kind": "pipeline", "id": "novel", "plugin": "transform/novel"},
            "description": "Supporting files for the novel contract.",
            "files": ["artifacts/novel/readme.txt"],
            "revision": 2,
            "revision_digest": "sha256:" + "b" * 64,
        },
    )

    deliverables = collect_notebook_deliverables(
        entries,
        outputs_dir=outputs,
        verification_status="ok",
        verification_issues=(),
        verification_records=(
            {"record_id": "novel/measurements", "status": "ok", "issues": []},
            {"record_id": "novel/supporting_files", "status": "ok", "issues": []},
        ),
    )

    assert deliverables.record_rows[0]["Contract"] == "novel.measurements.v1"
    assert deliverables.artifact_rows == (
        {
            "Kind": "file_bundle",
            "Record ID": "novel/supporting_files",
            "Producer": "pipeline:novel",
            "Plugin": "transform/novel",
            "Description": "Supporting files for the novel contract.",
            "File": "readme.txt",
            "Path": "artifacts/novel/readme.txt",
            "Revision": 2,
            "Revision digest": "sha256:" + "b" * 64,
            "Verification": "ok",
        },
    )

    loaded_record_ids: list[str] = []
    mo = _FakeMarimo()
    selector = build_notebook_deliverable_selector(mo, deliverables)
    assert "Data · novel/measurements · r3" in selector.options
    assert "Artifact · readme.txt · novel/supporting_files · r2" in selector.options
    dataframe_selector = type("Selector", (), {"value": "record:0"})()
    dataframe_viewport = render_notebook_deliverable_viewport(
        mo,
        deliverables,
        dataframe_selector,
        dataframe_loader=lambda record_id, revision, revision_digest: (
            loaded_record_ids.append(f"{record_id}@{revision}:{revision_digest}")
            or pd.DataFrame({"measurement": [1.0]})
        ),
        artifact_loader=lambda *_args: b"artifact",
    )
    artifact_selector = type("Selector", (), {"value": "artifact:0"})()
    artifact_viewport = render_notebook_deliverable_viewport(
        mo,
        deliverables,
        artifact_selector,
        dataframe_loader=lambda *_args: None,
        artifact_loader=lambda *_args: b"artifact",
    )

    assert loaded_record_ids == [f"novel/measurements@3:sha256:{'a' * 64}"]
    assert any(item.get("kind") == "table" for item in dataframe_viewport["items"] if isinstance(item, dict))
    assert any(
        item.get("kind") == "markdown" and "readme.txt" in item.get("text", "")
        for item in artifact_viewport["items"]
        if isinstance(item, dict)
    )


def test_deliverable_workbench_uses_one_selector_one_viewport_and_lazy_details(tmp_path: Path) -> None:
    plot = tmp_path / "plots" / "summary.png"
    plot.parent.mkdir()
    plot.write_bytes(b"image")
    mo = _FakeMarimo()
    deliverables = NotebookDeliverables(
        summary_rows=({"Deliverable": "Plot files", "Count": 1},),
        record_rows=(),
        plot_rows=(
            {
                "Record ID": "plot:summary",
                "File": "summary.png",
                "Path": "plots/summary.png",
                "Description": "Primary summary.",
                "Revision": 1,
                "Revision digest": "sha256:" + "a" * 64,
                "Verification": "ok",
            },
        ),
        export_rows=(),
        issue_rows=(),
        verification_status="ok",
    )

    selector = build_notebook_deliverable_selector(mo, deliverables)
    viewport = render_notebook_deliverable_viewport(
        mo,
        deliverables,
        selector,
        dataframe_loader=lambda *_args: None,
        artifact_loader=lambda *_args: plot.read_bytes(),
    )

    assert selector.label == "Deliverable"
    assert viewport["kind"] == "vstack"
    assert any(item.get("kind") == "image" for item in viewport["items"] if isinstance(item, dict))
    assert mo.accordion_calls[-1]["multiple"] is False
    assert mo.accordion_calls[-1]["lazy"] is True


def test_deliverable_workbench_previews_registered_pdf_plots_in_the_same_viewport(tmp_path: Path) -> None:
    plot = tmp_path / "plots" / "summary.pdf"
    plot.parent.mkdir()
    plot.write_bytes(b"%PDF-1.4\n")
    mo = _FakeMarimo()
    deliverables = NotebookDeliverables(
        summary_rows=({"Deliverable": "Plot files", "Count": 1},),
        record_rows=(),
        plot_rows=(
            {
                "Record ID": "plot:summary",
                "File": "summary.pdf",
                "Path": "plots/summary.pdf",
                "Description": "Primary summary.",
                "Revision": 1,
                "Revision digest": "sha256:" + "a" * 64,
                "Verification": "ok",
            },
        ),
        export_rows=(),
        issue_rows=(),
        verification_status="ok",
    )

    selector = build_notebook_deliverable_selector(mo, deliverables)
    viewport = render_notebook_deliverable_viewport(
        mo,
        deliverables,
        selector,
        dataframe_loader=lambda *_args: None,
        artifact_loader=lambda *_args: plot.read_bytes(),
    )

    preview = next(item for item in viewport["items"] if isinstance(item, dict) and item.get("kind") == "pdf")
    assert preview == {
        "kind": "pdf",
        "source": preview["source"],
        "width": "100%",
        "height": "70vh",
    }
    assert preview["source"].getvalue() == plot.read_bytes()


def test_deliverable_workbench_rejects_stale_selector_without_fallback(tmp_path: Path) -> None:
    mo = _FakeMarimo()
    deliverables = NotebookDeliverables(
        summary_rows=(),
        record_rows=(),
        plot_rows=(
            {
                "Record ID": "plot:summary",
                "File": "summary.png",
                "Path": "plots/summary.png",
                "Revision": 1,
                "Revision digest": "sha256:" + "a" * 64,
                "Verification": "ok",
            },
        ),
        export_rows=(),
        issue_rows=(),
        verification_status="ok",
    )

    with pytest.raises(ValueError, match="no longer exists.*refresh"):
        render_notebook_deliverable_viewport(
            mo,
            deliverables,
            type("Selector", (), {"value": "plot:stale"})(),
            dataframe_loader=lambda *_args: None,
            artifact_loader=lambda *_args: b"image",
        )


def test_deliverable_workbench_blocks_unverified_content_and_redacts_loader_errors(tmp_path: Path) -> None:
    mo = _FakeMarimo()
    private_path = tmp_path / "private-study" / "summary.png"
    row = {
        "Record ID": "plot:summary",
        "File": "summary.png",
        "Path": "plots/summary.png",
        "Revision": 1,
        "Revision digest": "sha256:" + "a" * 64,
        "Verification": "failed",
    }
    deliverables = NotebookDeliverables(
        summary_rows=(),
        record_rows=(),
        plot_rows=(row,),
        export_rows=(),
        issue_rows=(
            {
                "Scope": "record",
                "Record ID": "plot:summary",
                "Status": "failed",
                "Code": "artifact.digest_mismatch",
                "Field": "files.plots/summary.png",
                "Action": "Rerun the plot.",
                "Retryable": "no",
            },
        ),
        verification_status="failed",
    )
    loader_calls = []
    viewport = render_notebook_deliverable_viewport(
        mo,
        deliverables,
        type("Selector", (), {"value": "plot:0"})(),
        dataframe_loader=lambda *_args: None,
        artifact_loader=lambda *_args: loader_calls.append(True) or private_path.read_bytes(),
    )

    assert loader_calls == []
    assert "verification status is `failed`" in str(viewport)
    assert str(tmp_path) not in str(viewport)

    record_blocked = NotebookDeliverables(**{**deliverables.__dict__, "verification_status": "ok", "issue_rows": ()})
    blocked_calls = []
    record_blocked_viewport = render_notebook_deliverable_viewport(
        mo,
        record_blocked,
        type("Selector", (), {"value": "plot:0"})(),
        dataframe_loader=lambda *_args: None,
        artifact_loader=lambda *_args: blocked_calls.append(True) or b"image",
    )
    assert blocked_calls == []
    assert "selected record status is `failed`" in str(record_blocked_viewport)

    verified = NotebookDeliverables(
        **{
            **deliverables.__dict__,
            "verification_status": "ok",
            "issue_rows": (),
            "plot_rows": ({**row, "Verification": "ok"},),
        }
    )
    redacted = render_notebook_deliverable_viewport(
        mo,
        verified,
        type("Selector", (), {"value": "plot:0"})(),
        dataframe_loader=lambda *_args: None,
        artifact_loader=lambda *_args: (_ for _ in ()).throw(RuntimeError(str(private_path))),
    )
    assert "Could not verify and load" in str(redacted)
    assert str(tmp_path) not in str(redacted)
