from __future__ import annotations

from typing import Any


def resolve_pool_sets_arg(
    *,
    pool_sets: str | list[str] | list[dict[str, list[str]]] | None,
    group_on: str | None,
    groups: dict[str, Any] | None,
    allow_reference_lists: bool = False,
) -> list[dict[str, list[str]]] | None:
    if pool_sets is None:
        return None
    if isinstance(pool_sets, list) and (not pool_sets or isinstance(pool_sets[0], dict)):
        return pool_sets  # type: ignore[return-value]

    refs: list[str]
    if isinstance(pool_sets, str):
        refs = [pool_sets]
    elif isinstance(pool_sets, list) and allow_reference_lists:
        refs = [str(item) for item in pool_sets]
    else:
        raise ValueError("pool_sets must be a list[dict] or '<column>:<set>' reference")

    resolved: list[dict[str, list[str]]] = []
    for ref in refs:
        if ":" in ref:
            column, set_name = [segment.strip() for segment in ref.split(":", 1)]
        else:
            if not group_on:
                raise ValueError("pool_sets reference without group_on; use '<column>:<set>'")
            column, set_name = str(group_on), ref.strip()
        category = (groups or {}).get(column)
        if not isinstance(category, dict) or set_name not in category:
            options = ", ".join(sorted((category or {}).keys())) if isinstance(category, dict) else "—"
            raise ValueError(
                f"Unknown pool_sets reference '{ref}'. "
                f"Define it under semantics.groups.{column}.{set_name} in config. "
                f"(available for {column!r}: {options})"
            )
        sets_list = category[set_name]
        if not isinstance(sets_list, list):
            raise ValueError(f"semantics.groups.{column}.{set_name} must be a list of single-key dict objects")
        resolved.extend(sets_list)
    return resolved


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
