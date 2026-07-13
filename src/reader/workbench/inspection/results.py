from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from reader.errors import RegistryError
from reader.workbench.records import DataFrameArtifactRecord, FileBundleRecord, record_to_dict

from .common import format_relative_path


def record_detail_text(record, *, base: Path) -> str:
    if isinstance(record, DataFrameArtifactRecord):
        return f"{record.contract_id} • {format_relative_path(record.path, base=base)}"
    if not record.files:
        return "—"
    if len(record.files) == 1:
        return format_relative_path(record.files[0], base=base)
    parents = {format_relative_path(path.parent, base=base) for path in record.files}
    if len(parents) == 1:
        return f"{len(record.files)} files • {next(iter(parents))}"
    first = format_relative_path(record.files[0], base=base)
    return f"{len(record.files)} files • {len(parents)} directories • first: {first}"


def record_payload(
    record,
    *,
    outputs_dir: Path,
    runtime=None,
    revision_count: int | None = None,
) -> dict[str, object]:
    payload = record_to_dict(record, outputs_dir=outputs_dir)
    payload["producer_label"] = f"{record.producer.kind}:{record.producer.id}"
    if isinstance(record, FileBundleRecord) or runtime is not None:
        payload["description"] = record_description(record, runtime=runtime)
    if revision_count is not None:
        payload["revision_count"] = revision_count
    return payload


def record_description(record, *, runtime) -> str:
    if isinstance(record, FileBundleRecord):
        return record.description
    plugin_id = record.producer.plugin
    if not plugin_id:
        return "Description unavailable because the record has no plugin id."
    try:
        return runtime.plugins.resolve_descriptor(plugin_id).summary
    except RegistryError:
        return f"Description unavailable because plugin {plugin_id!r} is not registered."


def record_entries_payload(
    *,
    store,
    outputs_dir: Path,
    runtime=None,
    include_history: bool = False,
) -> list[dict[str, object]]:
    latest_records = store.iter_latest_records()
    if not include_history:
        return [record_payload(record, outputs_dir=outputs_dir, runtime=runtime) for record in latest_records]
    revision_counts = store.revision_counts(record.record_id for record in latest_records)
    return [
        record_payload(
            record,
            outputs_dir=outputs_dir,
            runtime=runtime,
            revision_count=revision_counts[record.record_id],
        )
        for record in latest_records
    ]


def record_summary_payload(
    *,
    latest_records,
    revision_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    kind_counts = Counter(record.kind for record in latest_records)
    producer_counts = Counter(f"{record.producer.kind}:{record.producer.id}" for record in latest_records)
    return {
        "records": len(latest_records),
        "history": {
            "included": revision_counts is not None,
            "revisions": sum(revision_counts.values()) if revision_counts is not None else None,
        },
        "by_kind": dict(sorted(kind_counts.items())),
        "by_producer": dict(sorted(producer_counts.items())),
    }


def record_catalog_payload(
    *,
    experiment: dict[str, object],
    store,
    outputs_dir: Path,
    runtime=None,
    include_history: bool = False,
) -> dict[str, object]:
    latest_records = store.iter_latest_records()
    revision_counts = store.revision_counts(record.record_id for record in latest_records) if include_history else None
    return {
        "experiment": deepcopy(experiment),
        "catalog": {
            "path": str(store.records_path),
            "outputs_root": str(outputs_dir),
        },
        "selection": {
            "include_history": include_history,
        },
        "summary": record_summary_payload(latest_records=latest_records, revision_counts=revision_counts),
        "records": [
            record_payload(
                record,
                outputs_dir=outputs_dir,
                runtime=runtime,
                revision_count=(revision_counts or {}).get(record.record_id),
            )
            for record in latest_records
        ],
    }
