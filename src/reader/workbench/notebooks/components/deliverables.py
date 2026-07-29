from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from reader.errors import RecordError
from reader.runtime import builtin_runtime
from reader.workbench.records import DataFrameArtifactRecord, FileBundleRecord


@dataclass(frozen=True)
class NotebookDeliverables:
    summary_rows: tuple[dict[str, Any], ...]
    record_rows: tuple[dict[str, Any], ...]
    plot_rows: tuple[dict[str, Any], ...]
    export_rows: tuple[dict[str, Any], ...]
    notebook_artifact_rows: tuple[dict[str, Any], ...]
    notebook_rows: tuple[dict[str, Any], ...]
    issue_rows: tuple[dict[str, Any], ...]


def build_notebook_deliverable_selector(mo: Any, deliverables: NotebookDeliverables) -> Any | None:
    options = _deliverable_options(deliverables)
    if not options:
        return None
    labels = {label: key for key, label, _ in options}
    return mo.ui.dropdown(
        options=labels,
        value=next(iter(labels)),
        label="Deliverable",
        full_width=True,
    )


def render_notebook_deliverable_viewport(
    mo: Any,
    deliverables: NotebookDeliverables,
    selector: Any | None,
    *,
    outputs_dir: Path,
) -> Any:
    options = {key: (label, row) for key, label, row in _deliverable_options(deliverables)}
    if selector is None or not options:
        primary = mo.md("No deliverables are registered yet. Run the experiment, then refresh this notebook.")
        selected_details = mo.md("No deliverable is selected.")
    else:
        selected = selector.value
        label, row = options.get(selected, next(iter(options.values())))
        primary = _render_deliverable_preview(mo, row, outputs_dir=outputs_dir)
        selected_details = _render_table(mo, (row,), "No metadata is available.")

    sections = {
        "Selected metadata": selected_details,
        "Summary": _render_table(mo, deliverables.summary_rows, "No deliverable summary is available."),
        "Plots": _render_table(mo, deliverables.plot_rows, "No plot files are registered yet."),
        "Exports": _render_table(mo, deliverables.export_rows, "No export files are registered yet."),
        "Records": _render_table(mo, deliverables.record_rows, "No dataframe records are registered yet."),
        "Notebook artifacts": _render_table(
            mo,
            deliverables.notebook_artifact_rows,
            "No notebook artifact files are registered yet.",
        ),
        "Notebooks": _render_table(mo, deliverables.notebook_rows, "No generated notebooks were found."),
    }
    if deliverables.issue_rows:
        sections["Readiness notes"] = _render_table(mo, deliverables.issue_rows, "No readiness notes.")
    controls = [] if selector is None else [selector]
    return mo.vstack(
        [
            mo.md("## Deliverables"),
            *controls,
            primary,
            mo.accordion(sections, multiple=False, lazy=True),
        ]
    )


def _deliverable_options(
    deliverables: NotebookDeliverables,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    groups = (
        ("plot", "Plot", deliverables.plot_rows),
        ("export", "Export", deliverables.export_rows),
        ("record", "Data", deliverables.record_rows),
        ("notebook_artifact", "Notebook artifact", deliverables.notebook_artifact_rows),
        ("notebook", "Notebook", deliverables.notebook_rows),
    )
    options: list[tuple[str, str, dict[str, Any]]] = []
    used_labels: set[str] = set()
    for kind, prefix, rows in groups:
        for index, row in enumerate(rows):
            identity = str(row.get("Record ID") or row.get("File") or row.get("Notebook") or row.get("Path"))
            label = f"{prefix} · {identity}"
            if label in used_labels:
                label = f"{label} · {index + 1}"
            used_labels.add(label)
            options.append((f"{kind}:{index}", label, row))
    return tuple(options)


def _render_deliverable_preview(mo: Any, row: dict[str, Any], *, outputs_dir: Path) -> Any:
    raw_path = row.get("Path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return _render_table(mo, (row,), "No preview is available.")
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else Path(outputs_dir) / candidate
    try:
        path = path.resolve(strict=True)
        path.relative_to(Path(outputs_dir).resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return mo.md("The selected deliverable is missing or resolves outside this experiment's outputs.")
    suffix = path.suffix.lower()
    description = str(row.get("Description") or row.get("Record ID") or path.name)
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
        return mo.image(str(path), alt=description)
    if suffix == ".parquet":
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            return mo.md(f"Could not preview `{path.name}`: `{exc}`")
        return mo.ui.table(frame.head(200), page_size=min(12, max(1, len(frame))))
    if suffix == ".csv":
        try:
            frame = pd.read_csv(path, nrows=200)
        except Exception as exc:
            return mo.md(f"Could not preview `{path.name}`: `{exc}`")
        return mo.ui.table(frame, page_size=min(12, max(1, len(frame))))
    return mo.md(f"**{path.name}**  \n{description}  \n`{path}`")


def collect_notebook_deliverables(outputs_dir: Path, *, notebooks_dir: Path | None = None) -> NotebookDeliverables:
    outputs = Path(outputs_dir).expanduser().resolve()
    notebooks = Path(notebooks_dir).expanduser().resolve() if notebooks_dir is not None else outputs / "notebooks"
    issue_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    notebook_artifact_rows: list[dict[str, Any]] = []

    runtime = builtin_runtime()
    try:
        store = runtime.record_store(outputs, create=False)
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
            elif record.producer.kind == "notebook":
                notebook_artifact_rows.extend(_file_rows(record, outputs_dir=outputs))

    notebook_rows = _notebook_rows(notebooks, outputs_dir=outputs)
    summary_rows = (
        {"Deliverable": "Dataframe records", "Count": len(record_rows)},
        {"Deliverable": "Plot files", "Count": len(plot_rows)},
        {"Deliverable": "Export files", "Count": len(export_rows)},
        {"Deliverable": "Notebook artifact files", "Count": len(notebook_artifact_rows)},
        {"Deliverable": "Generated notebooks", "Count": len(notebook_rows)},
    )
    return NotebookDeliverables(
        summary_rows=summary_rows,
        record_rows=tuple(record_rows),
        plot_rows=tuple(plot_rows),
        export_rows=tuple(export_rows),
        notebook_artifact_rows=tuple(notebook_artifact_rows),
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
    descriptions_by_path = {item.path: item.description for item in record.path_descriptions}
    bundle_description = record.description or "Description unavailable in this record."
    rows = []
    for path in record.files:
        description = (
            descriptions_by_path.get(path, bundle_description)
            if record.producer.kind in {"plot", "notebook"}
            else bundle_description
        )
        rows.append(
            {
                "Record ID": record.record_id,
                "Producer": _producer_label(record),
                "Plugin": record.producer.plugin or "",
                "Description": description,
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
