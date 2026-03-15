from __future__ import annotations

from pathlib import Path
from typing import Any

from reader.core.errors import RecordError

from .store import RecordStore


def _parse_step_dir(step_dir: str) -> tuple[str, str, str]:
    base = step_dir.split("__r")[0]
    if "." in base:
        step_id, plugin_key = base.split(".", 1)
    else:
        step_id, plugin_key = base, ""
    return base, step_id, plugin_key


def discover_dataframe_records(outputs_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str], str, str]:
    record_info: dict[str, dict[str, Any]] = {}
    record_note = ""
    record_warning = ""
    artifacts_dir = outputs_dir / "artifacts"
    records_path = outputs_dir / "manifests" / "records.json"

    def _register(label: str, *, path: Path, source: str, step_dir: str | None, record_id: str) -> None:
        display = label or record_id or (step_dir or path.stem)
        if display in record_info:
            display = f"{display}:{record_id}"
        _, step_id, plugin_key = _parse_step_dir(step_dir or "")
        record_info[display] = {
            "path": path,
            "record_id": record_id,
            "step_dir": step_dir or path.parent.name,
            "step_id": step_id or "",
            "plugin_key": plugin_key or "",
            "source": source,
            "base_label": label or record_id,
        }

    if records_path.exists():
        try:
            store = RecordStore(outputs_dir, create=False)
            for record in store.iter_latest_records(kind="dataframe_artifact"):
                _register(
                    record.record_id,
                    path=record.path,
                    source="catalog",
                    step_dir=record.path.parent.name,
                    record_id=record.record_id,
                )
            if not record_info:
                record_note = "No dataframe records listed in outputs/manifests/records.json."
        except RecordError as exc:
            record_note = f"Failed to read records.json: {exc}"

    if not record_info:
        if not artifacts_dir.exists():
            if not record_note:
                record_note = "No outputs/artifacts directory found. Run `reader run` first."
        else:
            for path in sorted(artifacts_dir.rglob("*.parquet")):
                step_dir = path.parent.name
                base_label, _, _ = _parse_step_dir(step_dir)
                _register(
                    base_label or path.stem,
                    path=path,
                    source="scan",
                    step_dir=step_dir,
                    record_id=path.stem,
                )
            if not record_info and not record_note:
                record_note = "No dataframe records found yet. Run `reader run` first."

    labels = sorted(record_info)
    if any(info.get("source") == "scan" for info in record_info.values()):
        record_warning = (
            "Warning: dataset list was built by scanning outputs/artifacts because "
            "outputs/manifests/records.json was missing, unreadable, or incomplete. "
            "Run `reader run` to regenerate the canonical record catalog."
        )
    return record_info, labels, record_note, record_warning
