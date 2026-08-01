from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from reader_workbench.errors import RegistryError
from reader_workbench.workbench.records import (
    DataFrameArtifactRecord,
    FileBundleRecord,
    record_revision_digest,
    record_to_dict,
)

from .common import format_relative_path


def _file_paths_detail_text(paths: list[str]) -> str:
    if not paths:
        return "—"
    if len(paths) == 1:
        return paths[0]
    parents = {str(Path(path).parent) for path in paths}
    if len(parents) == 1:
        return f"{len(paths)} files • {next(iter(parents))}"
    return f"{len(paths)} files • {len(parents)} directories • first: {paths[0]}"


def record_detail_text(record, *, base: Path) -> str:
    if isinstance(record, DataFrameArtifactRecord):
        return f"{record.contract_id} • {format_relative_path(record.path, base=base)}"
    return _file_paths_detail_text([format_relative_path(path, base=base) for path in record.files])


def record_payload_detail_text(record: dict[str, object]) -> str:
    if record.get("kind") == "dataframe_artifact":
        contract_id = str(record.get("contract_id") or "")
        path = str(record.get("path") or "")
        return " • ".join(value for value in (contract_id, path) if value) or "—"
    files = record.get("files")
    if not isinstance(files, list):
        return "—"
    return _file_paths_detail_text([str(path) for path in files if path])


def record_payload(
    record,
    *,
    outputs_dir: Path,
    runtime=None,
    revision: int,
    revision_count: int | None = None,
) -> dict[str, object]:
    payload = record_to_dict(record, outputs_dir=outputs_dir)
    payload["revision"] = revision
    payload["revision_digest"] = record_revision_digest(record, outputs_dir=outputs_dir)
    payload["producer_label"] = f"{record.producer.kind}:{record.producer.id}"
    if isinstance(record, FileBundleRecord) or runtime is not None:
        payload["description"] = record_description(record, runtime=runtime)
    if revision_count is not None:
        payload["revision_count"] = revision_count
    return payload


def record_description(record, *, runtime) -> str:
    if isinstance(record, FileBundleRecord):
        return record.description or "Description unavailable in this record."
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
    snapshot = store.catalog_snapshot()
    latest_records = snapshot.latest_records
    if not include_history:
        return [
            record_payload(
                record,
                outputs_dir=outputs_dir,
                runtime=runtime,
                revision=snapshot.revision_counts[record.record_id],
            )
            for record in latest_records
        ]
    revision_counts = snapshot.revision_counts
    return [
        record_payload(
            record,
            outputs_dir=outputs_dir,
            runtime=runtime,
            revision=revision_counts[record.record_id],
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
    current_config_digest: str | None = None,
    declared_record_ids: frozenset[str] = frozenset(),
) -> dict[str, object]:
    snapshot = store.catalog_snapshot(
        current_config_digest=current_config_digest if not include_history else None,
        current_record_ids=(declared_record_ids or None) if not include_history else None,
    )
    latest_records = snapshot.latest_records
    revision_counts = snapshot.revision_counts if include_history else None
    return {
        "experiment": deepcopy(experiment),
        "catalog": {
            "path": str(store.records_path),
            "outputs_root": str(outputs_dir),
            "schema_version": snapshot.schema_version,
            "provenance_epoch_id": snapshot.provenance_epoch_id,
            "active_invocation_ledger": str(snapshot.active_invocation_ledger),
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
                revision=snapshot.revision_counts[record.record_id],
                revision_count=(revision_counts or {}).get(record.record_id),
            )
            for record in latest_records
        ],
    }
