from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reader.errors import RecordError
from reader.runtime import builtin_runtime
from reader.workbench.records import DataFrameArtifactRecord, FileBundleRecord


@dataclass(frozen=True)
class NotebookDeliverables:
    summary_rows: tuple[dict[str, Any], ...]
    record_rows: tuple[dict[str, Any], ...]
    plot_rows: tuple[dict[str, Any], ...]
    export_rows: tuple[dict[str, Any], ...]
    notebook_rows: tuple[dict[str, Any], ...]
    issue_rows: tuple[dict[str, Any], ...]


def render_notebook_deliverables_panel(mo: Any, deliverables: NotebookDeliverables) -> Any:
    sections = {
        "Summary": _render_table(mo, deliverables.summary_rows, "No deliverable summary is available."),
        "Plots": _render_table(mo, deliverables.plot_rows, "No plot files are registered yet."),
        "Exports": _render_table(mo, deliverables.export_rows, "No export files are registered yet."),
        "Records": _render_table(mo, deliverables.record_rows, "No dataframe records are registered yet."),
        "Notebooks": _render_table(mo, deliverables.notebook_rows, "No generated notebooks were found."),
    }
    if deliverables.issue_rows:
        sections["Readiness notes"] = _render_table(mo, deliverables.issue_rows, "No readiness notes.")
    return mo.vstack(
        [
            mo.md("## Outputs and deliverables"),
            mo.accordion(sections, multiple=True, lazy=True),
        ]
    )


def collect_notebook_deliverables(outputs_dir: Path, *, notebooks_dir: Path | None = None) -> NotebookDeliverables:
    outputs = Path(outputs_dir).expanduser().resolve()
    notebooks = Path(notebooks_dir).expanduser().resolve() if notebooks_dir is not None else outputs / "notebooks"
    issue_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []

    try:
        store = builtin_runtime().record_store(outputs, create=False)
        records = store.iter_latest_records()
    except RecordError as exc:
        records = ()
        issue_rows.append({"Surface": "records", "Issue": str(exc)})

    for record in records:
        if isinstance(record, DataFrameArtifactRecord):
            record_rows.append(_dataframe_row(record, outputs_dir=outputs))
        elif isinstance(record, FileBundleRecord):
            if record.producer.kind == "plot":
                plot_rows.extend(_file_rows(record, outputs_dir=outputs))
            elif record.producer.kind == "export":
                export_rows.extend(_file_rows(record, outputs_dir=outputs))

    notebook_rows = _notebook_rows(notebooks, outputs_dir=outputs)
    summary_rows = (
        {"Deliverable": "Dataframe records", "Count": len(record_rows)},
        {"Deliverable": "Plot files", "Count": len(plot_rows)},
        {"Deliverable": "Export files", "Count": len(export_rows)},
        {"Deliverable": "Generated notebooks", "Count": len(notebook_rows)},
    )
    return NotebookDeliverables(
        summary_rows=summary_rows,
        record_rows=tuple(record_rows),
        plot_rows=tuple(plot_rows),
        export_rows=tuple(export_rows),
        notebook_rows=tuple(notebook_rows),
        issue_rows=tuple(issue_rows),
    )


def _render_table(mo: Any, rows: tuple[dict[str, Any], ...], empty_text: str) -> Any:
    rows_list = list(rows)
    if not rows_list:
        return mo.md(empty_text)
    return mo.ui.table(rows_list, page_size=min(12, max(1, len(rows_list))))


def _dataframe_row(record: DataFrameArtifactRecord, *, outputs_dir: Path) -> dict[str, Any]:
    return {
        "Record ID": record.record_id,
        "Producer": _producer_label(record),
        "Plugin": record.producer.plugin or "",
        "Contract": record.contract_id,
        "Path": _display_path(record.path, outputs_dir=outputs_dir),
        "Exists": _exists_label(record.path),
    }


def _file_rows(record: FileBundleRecord, *, outputs_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in record.files:
        rows.append(
            {
                "Record ID": record.record_id,
                "Producer": _producer_label(record),
                "Plugin": record.producer.plugin or "",
                "File": path.name,
                "Path": _display_path(path, outputs_dir=outputs_dir),
                "Exists": _exists_label(path),
            }
        )
    return rows


def _notebook_rows(notebooks_dir: Path, *, outputs_dir: Path) -> tuple[dict[str, Any], ...]:
    if not notebooks_dir.exists():
        return ()
    rows = []
    for path in sorted(notebooks_dir.glob("*.py")):
        rows.append(
            {
                "Notebook": path.name,
                "Path": _display_path(path, outputs_dir=outputs_dir),
                "Exists": _exists_label(path),
            }
        )
    return tuple(rows)


def _producer_label(record: DataFrameArtifactRecord | FileBundleRecord) -> str:
    return f"{record.producer.kind}:{record.producer.id}"


def _display_path(path: Path, *, outputs_dir: Path) -> str:
    try:
        return str(path.relative_to(outputs_dir))
    except ValueError:
        return str(path)


def _exists_label(path: Path) -> str:
    return "yes" if path.exists() else "no"
