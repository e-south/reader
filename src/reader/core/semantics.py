from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolvedPlotPartition:
    group_by: str | None
    collection_items: list[dict[str, list[str]]] | None
    match: str


def resolve_assay_collection_ref(
    *,
    ref: str,
    assay: dict[str, Any] | None,
) -> dict[str, Any]:
    collections = (assay or {}).get("collections") or {}
    collection_ref = str(ref).strip()
    if not collection_ref:
        raise ValueError("collection_ref must be a non-empty string")
    spec = collections.get(collection_ref)
    if spec is None:
        options = ", ".join(sorted(collections)) if isinstance(collections, dict) else "—"
        raise ValueError(
            f"Unknown collection_ref '{collection_ref}'. "
            f"Define it under assay.collections.{collection_ref}. (available: {options})"
        )
    if hasattr(spec, "model_dump"):
        spec = spec.model_dump()
    if not isinstance(spec, dict):
        raise ValueError(f"assay.collections.{collection_ref} must resolve to a mapping")
    column = spec.get("column")
    if not isinstance(column, str) or not column.strip():
        raise ValueError(f"assay.collections.{collection_ref}.column must be a non-empty string")
    items = spec.get("items")
    if not isinstance(items, dict) or not items:
        raise ValueError(f"assay.collections.{collection_ref}.items must be a non-empty mapping")

    normalized_items: list[dict[str, list[str]]] = []
    for label, values in items.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"assay.collections.{collection_ref}.items.{label} must be a non-empty list")
        normalized_items.append({str(label): [str(value) for value in values]})

    return {"column": str(column), "items": normalized_items}


def resolve_plot_partition(
    *,
    partition: dict[str, Any] | Any | None,
    assay: dict[str, Any] | None,
) -> ResolvedPlotPartition:
    if partition is None:
        return ResolvedPlotPartition(group_by=None, collection_items=None, match="exact")

    if hasattr(partition, "model_dump"):
        partition = partition.model_dump()
    if not isinstance(partition, dict):
        raise ValueError("partition must resolve to a mapping")

    group_by_raw = partition.get("by")
    if group_by_raw is not None and (not isinstance(group_by_raw, str) or not group_by_raw.strip()):
        raise ValueError("partition.by must be a non-empty string when provided")
    group_by = str(group_by_raw).strip() if isinstance(group_by_raw, str) else None

    collection_ref_raw = partition.get("collection_ref")
    if collection_ref_raw is not None and (not isinstance(collection_ref_raw, str) or not collection_ref_raw.strip()):
        raise ValueError("partition.collection_ref must be a non-empty string when provided")
    collection_ref = str(collection_ref_raw).strip() if isinstance(collection_ref_raw, str) else None

    match = partition.get("match", "exact")
    valid_match = {"exact", "contains", "startswith", "endswith", "regex"}
    if not isinstance(match, str) or match not in valid_match:
        raise ValueError(f"partition.match must be one of {sorted(valid_match)}")

    if collection_ref is None:
        return ResolvedPlotPartition(group_by=group_by, collection_items=None, match=match)

    collection = resolve_assay_collection_ref(ref=collection_ref, assay=assay)
    collection_column = collection["column"]
    if group_by is not None and group_by != collection_column:
        raise ValueError(
            f"partition.collection_ref '{collection_ref}' targets column {collection_column!r}, "
            f"but partition.by uses column {group_by!r}"
        )
    return ResolvedPlotPartition(
        group_by=collection_column,
        collection_items=collection["items"],
        match=match,
    )


def resolve_assay_label_refs(
    *,
    refs: list[str] | None,
    assay: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any]]]:
    if not refs:
        return []
    labels = (assay or {}).get("labels") or {}
    resolved: list[tuple[str, dict[str, Any]]] = []
    for raw_ref in refs:
        ref = str(raw_ref).strip()
        if not ref:
            raise ValueError("label refs must be non-empty strings")
        spec = labels.get(ref)
        if spec is None:
            options = ", ".join(sorted(labels)) if isinstance(labels, dict) else "—"
            raise ValueError(f"Unknown label ref '{ref}'. Define it under assay.labels.{ref}. (available: {options})")
        if hasattr(spec, "model_dump"):
            spec = spec.model_dump()
        if not isinstance(spec, dict):
            raise ValueError(f"assay.labels.{ref} must resolve to a mapping")
        resolved.append((ref, spec))
    return resolved


def resolve_assay_order_arg(
    *,
    order: list[str] | None,
    order_ref: str | None,
    column: str | None,
    assay: dict[str, Any] | None,
    arg_name: str,
) -> list[str] | None:
    if order is not None and order_ref is not None:
        raise ValueError(f"{arg_name} and {arg_name}_ref are mutually exclusive")
    if order is not None:
        if not order:
            raise ValueError(f"{arg_name} must not be empty when provided")
        return [str(item) for item in order]
    if order_ref is None:
        return None

    ref = str(order_ref).strip()
    if not ref:
        raise ValueError(f"{arg_name}_ref must be a non-empty string")
    orders = (assay or {}).get("orders") or {}
    spec = orders.get(ref)
    if spec is None:
        options = ", ".join(sorted(orders)) if isinstance(orders, dict) else "—"
        raise ValueError(f"Unknown {arg_name}_ref '{ref}'. Define it under assay.orders.{ref}. (available: {options})")
    if hasattr(spec, "model_dump"):
        spec = spec.model_dump()
    if not isinstance(spec, dict):
        raise ValueError(f"assay.orders.{ref} must resolve to a mapping")
    ref_column = spec.get("column")
    if column and ref_column and str(column) != str(ref_column):
        raise ValueError(f"{arg_name}_ref '{ref}' targets column {ref_column!r}, but plot uses column {column!r}")
    values = spec.get("values")
    if not isinstance(values, list) or any(isinstance(item, (dict, list)) for item in values):
        raise ValueError(f"assay.orders.{ref}.values must be a flat list of scalar labels")
    if not values:
        raise ValueError(f"assay.orders.{ref}.values must not be empty")
    return [str(item) for item in values]


def resolve_logic_map_ref(
    *,
    ref: str,
    assay: dict[str, Any] | None,
) -> dict[str, Any]:
    logic_maps = (assay or {}).get("logic_maps") or {}
    logic_ref = str(ref).strip()
    if not logic_ref:
        raise ValueError("logic_map_ref must be a non-empty string")
    spec = logic_maps.get(logic_ref)
    if spec is None:
        options = ", ".join(sorted(logic_maps)) if isinstance(logic_maps, dict) else "—"
        raise ValueError(
            f"Unknown logic_map_ref '{logic_ref}'. Define it under assay.logic_maps.{logic_ref}. (available: {options})"
        )
    if hasattr(spec, "model_dump"):
        spec = spec.model_dump()
    if not isinstance(spec, dict):
        raise ValueError(f"assay.logic_maps.{logic_ref} must resolve to a mapping")
    corners = spec.get("corners")
    if not isinstance(corners, dict) or set(corners) != {"00", "10", "01", "11"}:
        raise ValueError(f"assay.logic_maps.{logic_ref}.corners must have exactly 00/10/01/11")
    return {
        "column": str(spec.get("column")),
        "corners": {str(k): str(v) for k, v in corners.items()},
        "case_sensitive": bool(spec.get("case_sensitive", True)),
    }
