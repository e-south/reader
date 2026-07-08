from __future__ import annotations

from pathlib import Path

import pandas as pd

from reader.runtime import builtin_runtime
from reader.workbench.notebooks.components.deliverables import (
    NotebookDeliverables,
    collect_notebook_deliverables,
    render_notebook_deliverables_panel,
)


class _FakeUi:
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


def test_collect_notebook_deliverables_summarizes_records_plots_exports_and_notebooks(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    plots = outputs / "plots"
    exports = outputs / "exports"
    notebooks = outputs / "notebooks"
    plots.mkdir(parents=True)
    exports.mkdir(parents=True)
    notebooks.mkdir(parents=True)
    plot_file = plots / "summary.pdf"
    export_file = exports / "summary.xlsx"
    notebook_file = notebooks / "EDA_20260708.py"
    plot_file.write_text("plot", encoding="utf-8")
    export_file.write_text("export", encoding="utf-8")
    notebook_file.write_text("import marimo\n", encoding="utf-8")

    runtime = builtin_runtime()
    store = runtime.record_store(outputs, plots_subdir="plots", exports_subdir="exports")
    store.persist_dataframe(
        producer_id="summary",
        producer_plugin="transform/example",
        out_name="df",
        record_id="summary/df",
        df=pd.DataFrame({"value": [1]}),
        contract_id="none",
        inputs=[],
        config_digest="sha256:test",
    )
    store.append_file_bundle(
        producer_kind="plot",
        producer_id="plot_summary",
        producer_plugin="plot/example",
        record_id="plot:plot_summary",
        inputs=[],
        config_digest="sha256:test",
        files=[plot_file],
    )
    store.append_file_bundle(
        producer_kind="export",
        producer_id="export_summary",
        producer_plugin="export/example",
        record_id="export:export_summary",
        inputs=[],
        config_digest="sha256:test",
        files=[export_file],
    )

    deliverables = collect_notebook_deliverables(outputs, notebooks_dir=notebooks)

    assert {"Deliverable": "Dataframe records", "Count": 1} in deliverables.summary_rows
    assert {"Deliverable": "Plot files", "Count": 1} in deliverables.summary_rows
    assert {"Deliverable": "Export files", "Count": 1} in deliverables.summary_rows
    assert {"Deliverable": "Generated notebooks", "Count": 1} in deliverables.summary_rows
    assert deliverables.record_rows[0]["Record ID"] == "summary/df"
    assert deliverables.plot_rows[0]["Path"] == "plots/summary.pdf"
    assert deliverables.export_rows[0]["Path"] == "exports/summary.xlsx"
    assert deliverables.notebook_rows[0]["Path"] == "notebooks/EDA_20260708.py"


def test_render_notebook_deliverables_panel_uses_lazy_accordion() -> None:
    mo = _FakeMarimo()
    deliverables = NotebookDeliverables(
        summary_rows=({"Deliverable": "Plot files", "Count": 1},),
        record_rows=(),
        plot_rows=({"Path": "plots/summary.pdf"},),
        export_rows=(),
        notebook_rows=(),
        issue_rows=(),
    )

    panel = render_notebook_deliverables_panel(mo, deliverables)

    assert panel["kind"] == "vstack"
    assert mo.accordion_calls == [
        {
            "sections": {
                "Summary": {"kind": "table", "rows": [{"Deliverable": "Plot files", "Count": 1}], "page_size": 1},
                "Plots": {"kind": "table", "rows": [{"Path": "plots/summary.pdf"}], "page_size": 1},
                "Exports": {"kind": "markdown", "text": "No export files are registered yet."},
                "Records": {"kind": "markdown", "text": "No dataframe records are registered yet."},
                "Notebooks": {"kind": "markdown", "text": "No generated notebooks were found."},
            },
            "multiple": True,
            "lazy": True,
        }
    ]
