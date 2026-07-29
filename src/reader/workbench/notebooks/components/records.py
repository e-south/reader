from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def build_dataframe_record_catalog(
    entries: Iterable[Mapping[str, object]],
    *,
    catalog_exists: bool,
) -> tuple[dict[str, dict[str, Any]], list[str], str]:
    """Project public record entries into notebook selector metadata."""

    record_info: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.get("kind") != "dataframe_artifact":
            continue
        record_id = entry.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            continue
        producer = entry.get("producer")
        producer_info = producer if isinstance(producer, Mapping) else {}
        producer_id = producer_info.get("id")
        plugin_id = producer_info.get("plugin")
        record_info[record_id] = {
            "record_id": record_id,
            "step_id": producer_id if isinstance(producer_id, str) else "",
            "plugin_key": plugin_id if isinstance(plugin_id, str) else "",
            "created_at": str(entry.get("created_at") or ""),
        }

    labels = sorted(record_info)
    if labels:
        note = ""
    elif catalog_exists:
        note = "No dataframe records are registered. Run `uv run reader run` first."
    else:
        note = "No record catalog is registered. Run `uv run reader run` first."
    return record_info, labels, note


def select_default_dataframe_record(
    record_info: Mapping[str, Mapping[str, object]],
    record_labels: Sequence[str],
    *,
    pipeline_step_ids: Sequence[str] = (),
    preferred_record_ids: Sequence[str] = (),
) -> str | None:
    """Choose a stable catalog record without inspecting artifact paths."""

    if not record_labels:
        return None
    by_record_id = {
        str(info.get("record_id")): label
        for label, info in record_info.items()
        if isinstance(info.get("record_id"), str)
    }
    for record_id in preferred_record_ids:
        if record_id in by_record_id:
            return by_record_id[record_id]
    for step_id in reversed(tuple(pipeline_step_ids)):
        matches = sorted(label for label, info in record_info.items() if info.get("step_id") == step_id)
        if matches:
            return matches[0]
    return max(
        record_labels,
        key=lambda label: (str(record_info[label].get("created_at") or ""), label),
    )
