from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NotebookDeliverables:
    summary_rows: tuple[dict[str, Any], ...]
    record_rows: tuple[dict[str, Any], ...]
    plot_rows: tuple[dict[str, Any], ...]
    export_rows: tuple[dict[str, Any], ...]
    issue_rows: tuple[dict[str, Any], ...]
    verification_status: str
    artifact_rows: tuple[dict[str, Any], ...] = ()


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
    dataframe_loader: Callable[[str, int, str], Any],
    artifact_loader: Callable[[str, int, str, str], bytes],
) -> Any:
    options = {key: (label, row) for key, label, row in _deliverable_options(deliverables)}
    if selector is None or not options:
        primary = mo.md("No deliverables are registered yet. Run the experiment, then refresh this notebook.")
        selected_details = mo.md("No deliverable is selected.")
    else:
        selected = selector.value
        if selected not in options:
            raise ValueError("The selected deliverable no longer exists; refresh the notebook catalog.")
        _label, row = options[selected]
        selected_verification = str(row.get("Verification") or "unknown")
        if deliverables.verification_status != "ok" or (row.get("Record ID") and selected_verification != "ok"):
            primary = mo.md(
                "Preview unavailable because Reader verification status is "
                f"`{deliverables.verification_status}` and the selected record status is "
                f"`{selected_verification}`. Review readiness notes, repair the records, and refresh."
            )
        else:
            primary = _render_deliverable_preview(
                mo,
                row,
                dataframe_loader=dataframe_loader,
                artifact_loader=artifact_loader,
            )
        selected_details = _render_table(mo, (row,), "No metadata is available.")

    sections = {
        "Selected metadata": selected_details,
        "Summary": _render_table(mo, deliverables.summary_rows, "No deliverable summary is available."),
        "Plots": _render_table(mo, deliverables.plot_rows, "No plot files are registered yet."),
        "Exports": _render_table(mo, deliverables.export_rows, "No export files are registered yet."),
        "Records": _render_table(mo, deliverables.record_rows, "No dataframe records are registered yet."),
        "Artifacts": _render_table(mo, deliverables.artifact_rows, "No other artifact files are registered yet."),
    }
    if deliverables.issue_rows:
        sections["Readiness notes"] = _render_table(mo, deliverables.issue_rows, "No readiness notes.")
    controls = [] if selector is None else [selector]
    return mo.vstack(
        [
            mo.md("## Deliverables"),
            mo.md(f"**Verification:** `{deliverables.verification_status}`"),
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
        ("artifact", "Artifact", deliverables.artifact_rows),
    )
    options: list[tuple[str, str, dict[str, Any]]] = []
    used_labels: set[str] = set()
    for kind, prefix, rows in groups:
        for index, row in enumerate(rows):
            record_id = str(row.get("Record ID") or "")
            revision = row.get("Revision")
            filename = str(row.get("File") or "")
            if record_id and filename:
                identity = f"{filename} · {record_id} · r{revision}"
            elif record_id:
                identity = f"{record_id} · r{revision}"
            else:
                identity = str(row.get("Path"))
            label = f"{prefix} · {identity}"
            if label in used_labels:
                label = f"{label} · {index + 1}"
            used_labels.add(label)
            options.append((f"{kind}:{index}", label, row))
    return tuple(options)


def _render_deliverable_preview(
    mo: Any,
    row: dict[str, Any],
    *,
    dataframe_loader: Callable[[str, int, str], Any],
    artifact_loader: Callable[[str, int, str, str], bytes],
) -> Any:
    raw_path = row.get("Path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return _render_table(mo, (row,), "No preview is available.")
    if Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
        return mo.md("The selected deliverable does not have a safe experiment-relative path.")
    suffix = Path(raw_path).suffix.lower()
    description = str(row.get("Description") or row.get("Record ID") or Path(raw_path).name)
    record_id = row.get("Record ID")
    revision = row.get("Revision")
    revision_digest = row.get("Revision digest")
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
        try:
            content = artifact_loader(
                _required_record_id(record_id),
                _required_revision(revision),
                _required_revision_digest(revision_digest),
                raw_path,
            )
        except Exception:
            return mo.md("Could not verify and load the selected artifact. Refresh the catalog or rerun its producer.")
        return mo.image(content, alt=description)
    if suffix == ".pdf":
        try:
            content = artifact_loader(
                _required_record_id(record_id),
                _required_revision(revision),
                _required_revision_digest(revision_digest),
                raw_path,
            )
        except Exception:
            return mo.md("Could not verify and load the selected artifact. Refresh the catalog or rerun its producer.")
        return mo.pdf(BytesIO(content), width="100%", height="70vh")
    if row.get("Kind") == "dataframe_artifact":
        try:
            frame = dataframe_loader(
                _required_record_id(record_id),
                _required_revision(revision),
                _required_revision_digest(revision_digest),
            )
        except Exception:
            return mo.md("Could not verify and load the selected dataframe. Refresh the catalog or rerun its producer.")
        return mo.ui.table(frame.head(200), page_size=min(12, max(1, len(frame))))
    return mo.md(f"**{Path(raw_path).name}**  \n{description}  \n`{raw_path}`")


def _required_record_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("deliverable record id is missing")
    return value


def _required_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("deliverable revision is invalid")
    return value


def _required_revision_digest(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError("deliverable revision digest is invalid")
    return value


def collect_notebook_deliverables(
    entries: Iterable[Mapping[str, object]],
    *,
    outputs_dir: Path,
    verification_status: str,
    verification_issues: Iterable[Mapping[str, object]],
    verification_records: Iterable[Mapping[str, object]],
) -> NotebookDeliverables:
    outputs = Path(outputs_dir).expanduser().resolve()
    verification_issue_items = tuple(verification_issues)
    verification_record_items = tuple(verification_records)
    record_statuses = {
        str(item.get("record_id")): str(item.get("status") or "unknown")
        for item in verification_record_items
        if isinstance(item.get("record_id"), str)
    }
    record_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []

    for entry in entries:
        if entry.get("kind") == "dataframe_artifact":
            record_rows.append(
                _dataframe_row(
                    entry,
                    outputs_dir=outputs,
                    verification_status=record_statuses.get(str(entry.get("record_id")), "unknown"),
                )
            )
        elif entry.get("kind") == "file_bundle":
            producer = entry.get("producer")
            producer_info = producer if isinstance(producer, Mapping) else {}
            producer_kind = producer_info.get("kind")
            if producer_kind == "plot":
                plot_rows.extend(
                    _file_rows(
                        entry,
                        outputs_dir=outputs,
                        verification_status=record_statuses.get(str(entry.get("record_id")), "unknown"),
                    )
                )
            elif producer_kind == "export":
                export_rows.extend(
                    _file_rows(
                        entry,
                        outputs_dir=outputs,
                        verification_status=record_statuses.get(str(entry.get("record_id")), "unknown"),
                    )
                )
            else:
                artifact_rows.extend(
                    _file_rows(
                        entry,
                        outputs_dir=outputs,
                        verification_status=record_statuses.get(str(entry.get("record_id")), "unknown"),
                    )
                )

    summary_rows = (
        {"Deliverable": "Dataframe records", "Count": len(record_rows)},
        {"Deliverable": "Plot files", "Count": len(plot_rows)},
        {"Deliverable": "Export files", "Count": len(export_rows)},
        {"Deliverable": "Other artifact files", "Count": len(artifact_rows)},
    )
    return NotebookDeliverables(
        summary_rows=summary_rows,
        record_rows=tuple(record_rows),
        plot_rows=tuple(plot_rows),
        export_rows=tuple(export_rows),
        issue_rows=_readiness_rows(
            status=verification_status,
            issues=verification_issue_items,
            records=verification_record_items,
        ),
        verification_status=verification_status,
        artifact_rows=tuple(artifact_rows),
    )


def _render_table(mo: Any, rows: tuple[dict[str, Any], ...], empty_text: str) -> Any:
    rows_list = list(rows)
    if not rows_list:
        return mo.md(empty_text)
    return mo.ui.table(rows_list, page_size=min(12, max(1, len(rows_list))))


def _dataframe_row(
    record: Mapping[str, object],
    *,
    outputs_dir: Path,
    verification_status: str,
) -> dict[str, Any]:
    path = _entry_path(record.get("path"), outputs_dir=outputs_dir)
    return {
        "Kind": "dataframe_artifact",
        "Record ID": str(record.get("record_id") or ""),
        "Producer": _producer_label(record),
        "Plugin": _producer_value(record, "plugin"),
        "Contract": str(record.get("contract_id") or ""),
        "Path": _display_path(path, outputs_dir=outputs_dir),
        "Revision": record.get("revision"),
        "Revision digest": record.get("revision_digest"),
        "Verification": verification_status,
    }


def _file_rows(
    record: Mapping[str, object],
    *,
    outputs_dir: Path,
    verification_status: str,
) -> list[dict[str, Any]]:
    raw_descriptions = record.get("path_descriptions")
    descriptions_by_path = (
        {
            str(item.get("path")): str(item.get("description") or "")
            for item in raw_descriptions
            if isinstance(item, Mapping) and isinstance(item.get("path"), str)
        }
        if isinstance(raw_descriptions, list)
        else {}
    )
    bundle_description = str(record.get("description") or "Description unavailable in this record.")
    producer_kind = _producer_value(record, "kind")
    raw_files = record.get("files")
    files = raw_files if isinstance(raw_files, list) else []
    rows = []
    for raw_path in files:
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = _entry_path(raw_path, outputs_dir=outputs_dir)
        description = (
            descriptions_by_path.get(raw_path, bundle_description) if producer_kind == "plot" else bundle_description
        )
        rows.append(
            {
                "Kind": "file_bundle",
                "Record ID": str(record.get("record_id") or ""),
                "Producer": _producer_label(record),
                "Plugin": _producer_value(record, "plugin"),
                "Description": description,
                "File": path.name,
                "Path": _display_path(path, outputs_dir=outputs_dir),
                "Revision": record.get("revision"),
                "Revision digest": record.get("revision_digest"),
                "Verification": verification_status,
            }
        )
    return rows


def _producer_label(record: Mapping[str, object]) -> str:
    return f"{_producer_value(record, 'kind')}:{_producer_value(record, 'id')}"


def _producer_value(record: Mapping[str, object], key: str) -> str:
    producer = record.get("producer")
    if not isinstance(producer, Mapping):
        return ""
    return str(producer.get(key) or "")


def _entry_path(raw_path: object, *, outputs_dir: Path) -> Path:
    path = Path(str(raw_path or ""))
    return path if path.is_absolute() else outputs_dir / path


def _display_path(path: Path, *, outputs_dir: Path) -> str:
    try:
        relative = path.relative_to(outputs_dir)
    except ValueError:
        return "[outside outputs]"
    if relative == Path(".") or ".." in relative.parts:
        return "[outside outputs]"
    return relative.as_posix()


def _readiness_rows(
    *,
    status: str,
    issues: Iterable[Mapping[str, object]],
    records: Iterable[Mapping[str, object]],
) -> tuple[dict[str, Any], ...]:
    rows = [_readiness_row(issue, scope="catalog", record_id="", status=status) for issue in issues]
    for record in records:
        record_id = str(record.get("record_id") or "")
        record_status = str(record.get("status") or "unknown")
        raw_issues = record.get("issues")
        if not isinstance(raw_issues, list):
            continue
        rows.extend(
            _readiness_row(issue, scope="record", record_id=record_id, status=record_status)
            for issue in raw_issues
            if isinstance(issue, Mapping)
        )
    return tuple(rows)


def _readiness_row(
    issue: Mapping[str, object],
    *,
    scope: str,
    record_id: str,
    status: str,
) -> dict[str, Any]:
    return {
        "Scope": scope,
        "Record ID": record_id,
        "Status": status,
        "Code": str(issue.get("code") or "unknown"),
        "Field": str(issue.get("field") or ""),
        "Action": str(issue.get("remediation") or "Inspect Reader verification output."),
        "Retryable": "yes" if issue.get("retryable") is True else "no",
    }
