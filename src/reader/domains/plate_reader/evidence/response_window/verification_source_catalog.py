"""Verify selected source-record claims against bundled Reader catalogs."""

from __future__ import annotations

import json
from pathlib import Path

from reader.domains.plate_reader.analysis.response_window.sources import ANNOTATED_CONTRACT


def verify_source_catalog(path: Path, digests: dict[object, object]) -> None:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("response-window bundled source record catalog is unreadable.") from exc
    latest = catalog.get("latest") if isinstance(catalog, dict) else None
    if not isinstance(latest, dict):
        raise ValueError("response-window bundled source record catalog lacks latest records.")
    for raw_record_id, digest in digests.items():
        record_id = str(raw_record_id)
        selected = latest.get(record_id)
        if not isinstance(selected, dict) or selected.get("record_id") != record_id:
            raise ValueError(f"response-window source record identity disagrees with bundled catalog: {record_id!r}.")
        if selected.get("contract_id") != ANNOTATED_CONTRACT:
            raise ValueError(f"response-window source record contract disagrees with bundled catalog: {record_id!r}.")
        if selected.get("content_digest") != digest:
            raise ValueError(f"response-window source record digest disagrees with bundled catalog: {record_id!r}.")


__all__ = ["verify_source_catalog"]
