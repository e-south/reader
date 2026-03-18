from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path


def _sorted_counter(values: list[str]) -> dict[str, int]:
    counts = Counter(value for value in values if value)
    return dict(sorted(counts.items()))


def inventory_summary_payload(entries: list[dict[str, object]]) -> dict[str, object]:
    statuses = [str(entry.get("status") or "unknown") for entry in entries]
    lifecycles = [str(entry.get("lifecycle") or "unknown") for entry in entries]
    readiness_states = [
        str(readiness.get("state"))
        for readiness in (entry.get("readiness") for entry in entries)
        if isinstance(readiness, dict) and readiness.get("state")
    ]
    protocols = [
        str(protocol)
        for protocol in (entry.get("protocol") for entry in entries)
        if isinstance(protocol, str) and protocol
    ]
    with_outputs = sum(1 for entry in entries if bool(entry.get("has_outputs")))
    payload = {
        "experiments": len(entries),
        "by_status": _sorted_counter(statuses),
        "by_lifecycle": _sorted_counter(lifecycles),
        "by_protocol": _sorted_counter(protocols),
        "outputs": {
            "with_outputs": with_outputs,
            "without_outputs": len(entries) - with_outputs,
        },
    }
    if readiness_states:
        payload["by_readiness"] = _sorted_counter(readiness_states)
    return payload


def inventory_surface_payload(
    *,
    root: Path,
    include_scaffolds: bool,
    details: bool,
    readiness: bool,
    protocol: str | None,
    status: str | None,
    lifecycle: str | None,
    experiments: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "catalog": {
            "kind": "experiments",
            "root": str(root),
        },
        "selection": {
            "include_scaffolds": include_scaffolds,
            "details": details,
            "readiness": readiness,
            "protocol": protocol,
            "status": status,
            "lifecycle": lifecycle,
        },
        "summary": inventory_summary_payload(experiments),
        "experiments": deepcopy(experiments),
    }
