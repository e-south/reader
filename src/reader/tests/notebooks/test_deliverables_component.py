from __future__ import annotations

from pathlib import Path

import pandas as pd

from reader.runtime import builtin_runtime
from reader.workbench.graph import ProvenanceInput, RecordRef
from reader.workbench.notebooks.components.deliverables import (
    NotebookDeliverables,
    build_notebook_deliverable_selector,
    collect_notebook_deliverables,
    render_notebook_deliverable_viewport,
)
from reader.workbench.notebooks.components.records import build_dataframe_record_catalog
from reader.workbench.records import PathDescription, record_to_dict


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


def test_collect_notebook_deliverables_summarizes_records_plots_exports_and_notebooks(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    plots = outputs / "plots"
    exports = outputs / "exports"
    notebooks = outputs / "notebooks"
    plots.mkdir(parents=True)
    exports.mkdir(parents=True)
    notebooks.mkdir(parents=True)
    plot_file = plots / "summary.pdf"
    second_plot_file = plots / "kinetics.pdf"
    export_file = exports / "summary.xlsx"
    notebook_file = notebooks / "EDA_20260708.py"
    plot_file.write_text("plot", encoding="utf-8")
    second_plot_file.write_text("plot", encoding="utf-8")
    export_file.write_text("export", encoding="utf-8")
    notebook_file.write_text("import marimo\n", encoding="utf-8")

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
    notebook_pdf = exports / "cytometry" / "cytometry_eda.pdf"
    notebook_pdf.parent.mkdir()
    notebook_pdf.write_text("pdf", encoding="utf-8")
    store.append_notebook_file_bundle(
        producer_id="cytometry_eda",
        producer_template="notebook/cytometry",
        record_id="notebook:cytometry_eda",
        inputs=store.capture_inputs([ProvenanceInput(label="events", ref=RecordRef(record_id="summary/df"))]),
        config_digest="sha256:test",
        producer_config_digest="sha256:notebook",
        files=[notebook_pdf],
        description="Interactive cytometry outputs.",
        path_descriptions=(PathDescription(path=notebook_pdf, description="Interactive cytometry EDA plot."),),
    )

    entries = tuple(record_to_dict(record, outputs_dir=outputs) for record in store.iter_latest_records())
    deliverables = collect_notebook_deliverables(
        entries,
        outputs_dir=outputs,
        notebooks_dir=notebooks,
    )

    assert {"Deliverable": "Dataframe records", "Count": 1} in deliverables.summary_rows
    assert {"Deliverable": "Plot files", "Count": 2} in deliverables.summary_rows
    assert {"Deliverable": "Export files", "Count": 1} in deliverables.summary_rows
    assert {"Deliverable": "Notebook artifact files", "Count": 1} in deliverables.summary_rows
    assert {"Deliverable": "Generated notebooks", "Count": 1} in deliverables.summary_rows
    assert deliverables.record_rows[0]["Record ID"] == "summary/df"
    plot_rows = {row["Path"]: row for row in deliverables.plot_rows}
    assert plot_rows["plots/summary.pdf"]["Description"] == "Endpoint summary by treatment."
    assert plot_rows["plots/kinetics.pdf"]["Description"] == "Reporter kinetics over assay time."
    assert deliverables.export_rows[0]["Path"] == "exports/summary.xlsx"
    assert deliverables.export_rows[0]["Description"] == "Write dataframe records to XLSX workbooks."
    notebook_row = deliverables.notebook_artifact_rows[0]
    assert notebook_row["Path"] == "exports/cytometry/cytometry_eda.pdf"
    assert notebook_row["Producer"] == "notebook:cytometry_eda"
    assert notebook_row["Description"] == "Interactive cytometry EDA plot."
    assert deliverables.notebook_rows[0]["Path"] == "notebooks/EDA_20260708.py"


def test_collect_notebook_deliverables_handles_empty_public_catalog(tmp_path: Path) -> None:
    deliverables = collect_notebook_deliverables((), outputs_dir=tmp_path / "outputs")

    assert deliverables.record_rows == ()
    assert deliverables.plot_rows == ()
    assert deliverables.export_rows == ()
    assert deliverables.issue_rows == ()
    assert deliverables.artifact_rows == ()


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
        },
        {
            "kind": "file_bundle",
            "record_id": "novel/supporting_files",
            "producer": {"kind": "pipeline", "id": "novel", "plugin": "transform/novel"},
            "description": "Supporting files for the novel contract.",
            "files": ["artifacts/novel/readme.txt"],
        },
    )

    record_info, record_labels, note = build_dataframe_record_catalog(entries, catalog_exists=True)
    deliverables = collect_notebook_deliverables(entries, outputs_dir=outputs)

    assert record_labels == ["novel/measurements"]
    assert record_info["novel/measurements"]["plugin_key"] == "transform/novel"
    assert note == ""
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
            "Exists": "yes",
        },
    )

    loaded_record_ids: list[str] = []
    mo = _FakeMarimo()
    selector = build_notebook_deliverable_selector(mo, deliverables)
    assert "Data · novel/measurements" in selector.options
    assert "Artifact · novel/supporting_files" in selector.options
    dataframe_selector = type("Selector", (), {"value": "record:0"})()
    dataframe_viewport = render_notebook_deliverable_viewport(
        mo,
        deliverables,
        dataframe_selector,
        outputs_dir=outputs,
        dataframe_loader=lambda record_id: (
            loaded_record_ids.append(record_id) or pd.DataFrame({"measurement": [1.0]})
        ),
    )
    artifact_selector = type("Selector", (), {"value": "artifact:0"})()
    artifact_viewport = render_notebook_deliverable_viewport(
        mo,
        deliverables,
        artifact_selector,
        outputs_dir=outputs,
        dataframe_loader=lambda _record_id: None,
    )

    assert loaded_record_ids == ["novel/measurements"]
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
            },
        ),
        export_rows=(),
        notebook_artifact_rows=(),
        notebook_rows=(),
        issue_rows=(),
    )

    selector = build_notebook_deliverable_selector(mo, deliverables)
    viewport = render_notebook_deliverable_viewport(
        mo,
        deliverables,
        selector,
        outputs_dir=tmp_path,
        dataframe_loader=lambda _record_id: None,
    )

    assert selector.label == "Deliverable"
    assert viewport["kind"] == "vstack"
    assert any(item.get("kind") == "image" for item in viewport["items"] if isinstance(item, dict))
    assert mo.accordion_calls[-1]["multiple"] is False
    assert mo.accordion_calls[-1]["lazy"] is True
