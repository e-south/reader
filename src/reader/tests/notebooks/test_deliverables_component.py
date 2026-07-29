from __future__ import annotations

import json
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
from reader.workbench.records import PathDescription


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

    deliverables = collect_notebook_deliverables(outputs, notebooks_dir=notebooks)

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


def test_collect_notebook_deliverables_reports_retired_record_schema_as_invalid(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    runtime = builtin_runtime()
    store = runtime.record_store(outputs)
    payload = {
        "schema_version": 3,
        "record_id": "plot:missing_descriptions",
        "kind": "file_bundle",
        "producer": {"kind": "plot", "id": "missing_descriptions", "plugin": "plot/time_series"},
        "created_at": "2026-07-10T00:00:00+00:00",
        "inputs": [],
        "config_digest": "sha256:missing-descriptions",
        "files": ["plots/missing_descriptions.png"],
        "description": "Bundle-level description only.",
    }
    catalog = {
        "schema_version": 3,
        "latest": {"plot:missing_descriptions": payload},
        "history": {"plot:missing_descriptions": [payload]},
    }
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    deliverables = collect_notebook_deliverables(outputs)

    assert deliverables.plot_rows == ()
    assert len(deliverables.issue_rows) == 1
    assert deliverables.issue_rows[0]["Surface"] == "records"
    assert "schema_version must be 5" in deliverables.issue_rows[0]["Issue"]


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
    viewport = render_notebook_deliverable_viewport(mo, deliverables, selector, outputs_dir=tmp_path)

    assert selector.label == "Deliverable"
    assert viewport["kind"] == "vstack"
    assert any(item.get("kind") == "image" for item in viewport["items"] if isinstance(item, dict))
    assert mo.accordion_calls[-1]["multiple"] is False
    assert mo.accordion_calls[-1]["lazy"] is True
