from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from reader.protocols.model import ProtocolBinding

ResourceKind = Literal["file", "directory"]


@dataclass(frozen=True)
class AnnotationLabelSpec:
    source: str
    values: dict[str, str] = field(default_factory=dict)
    output: str | None = None


@dataclass(frozen=True)
class ResolvedAnnotationLabelSpec:
    ref: str
    source: str
    output: str | None
    values: dict[str, str]


@dataclass(frozen=True)
class AnnotationLabels:
    by_id: dict[str, AnnotationLabelSpec] = field(default_factory=dict)

    def resolve(self, refs: list[str] | None = None) -> list[ResolvedAnnotationLabelSpec]:
        requested = list(self.by_id) if refs is None else refs
        if not requested:
            return []
        resolved: list[ResolvedAnnotationLabelSpec] = []
        for raw_ref in requested:
            ref = str(raw_ref).strip()
            if not ref:
                raise ValueError("label refs must be non-empty strings")
            spec = self.by_id.get(ref)
            if spec is None:
                raise ValueError(f"annotations.labels missing key '{ref}'")
            resolved.append(
                ResolvedAnnotationLabelSpec(
                    ref=ref,
                    source=spec.source,
                    output=spec.output,
                    values=dict(spec.values),
                )
            )
        return resolved


@dataclass(frozen=True)
class AnnotationOrderSpec:
    column: str
    values: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnnotationOrders:
    by_id: dict[str, AnnotationOrderSpec] = field(default_factory=dict)

    def resolve(
        self,
        *,
        order: list[str] | None,
        order_ref: str | None,
        column: str | None,
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
        spec = self.by_id.get(ref)
        if spec is None:
            options = ", ".join(sorted(self.by_id)) if self.by_id else "—"
            raise ValueError(
                f"Unknown {arg_name}_ref '{ref}'. Define it under annotations.orders.{ref}. (available: {options})"
            )
        if column and spec.column and str(column) != str(spec.column):
            raise ValueError(f"{arg_name}_ref '{ref}' targets column {spec.column!r}, but plot uses column {column!r}")
        if not spec.values:
            raise ValueError(f"annotations.orders.{ref}.values must not be empty")
        return [str(item) for item in spec.values]


@dataclass(frozen=True)
class AnnotationCollectionSpec:
    column: str
    items: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnotationCollections:
    by_id: dict[str, AnnotationCollectionSpec] = field(default_factory=dict)

    def resolve(self, *, ref: str) -> dict[str, Any]:
        collection_ref = str(ref).strip()
        if not collection_ref:
            raise ValueError("collection_ref must be a non-empty string")
        spec = self.by_id.get(collection_ref)
        if spec is None:
            options = ", ".join(sorted(self.by_id)) if self.by_id else "—"
            raise ValueError(
                f"Unknown collection_ref '{collection_ref}'. "
                f"Define it under annotations.collections.{collection_ref}. (available: {options})"
            )
        if not spec.items:
            raise ValueError(f"annotations.collections.{collection_ref}.items must be a non-empty mapping")
        normalized_items: list[dict[str, list[str]]] = []
        for label, values in spec.items.items():
            if not values:
                raise ValueError(f"annotations.collections.{collection_ref}.items.{label} must be a non-empty list")
            normalized_items.append({str(label): [str(value) for value in values]})
        return {"column": spec.column, "items": normalized_items}


@dataclass(frozen=True)
class LogicMapSpec:
    column: str
    corners: dict[str, str]
    case_sensitive: bool = True


@dataclass(frozen=True)
class ResolvedLogicMap:
    column: str
    corners: dict[str, str]
    case_sensitive: bool


@dataclass(frozen=True)
class LogicMaps:
    by_id: dict[str, LogicMapSpec] = field(default_factory=dict)

    def resolve(self, *, ref: str) -> ResolvedLogicMap:
        logic_ref = str(ref).strip()
        if not logic_ref:
            raise ValueError("logic_map_ref must be a non-empty string")
        spec = self.by_id.get(logic_ref)
        if spec is None:
            options = ", ".join(sorted(self.by_id)) if self.by_id else "—"
            raise ValueError(
                f"Unknown logic_map_ref '{logic_ref}'. Define it under annotations.logic_maps.{logic_ref}. (available: {options})"
            )
        if set(spec.corners) != {"00", "10", "01", "11"}:
            raise ValueError(f"annotations.logic_maps.{logic_ref}.corners must have exactly 00/10/01/11")
        return ResolvedLogicMap(
            column=spec.column,
            corners={str(key): str(value) for key, value in spec.corners.items()},
            case_sensitive=bool(spec.case_sensitive),
        )


@dataclass(frozen=True)
class ResolvedPlotPartition:
    group_by: str | None
    collection_items: list[dict[str, list[str]]] | None
    match: str


@dataclass(frozen=True)
class AnnotationSemantics:
    labels: AnnotationLabels = field(default_factory=AnnotationLabels)
    orders: AnnotationOrders = field(default_factory=AnnotationOrders)
    collections: AnnotationCollections = field(default_factory=AnnotationCollections)
    logic_maps: LogicMaps = field(default_factory=LogicMaps)

    def resolve_label_specs(self, refs: list[str] | None = None) -> list[ResolvedAnnotationLabelSpec]:
        return self.labels.resolve(refs)

    def resolve_order_arg(
        self,
        *,
        order: list[str] | None,
        order_ref: str | None,
        column: str | None,
        arg_name: str,
    ) -> list[str] | None:
        return self.orders.resolve(order=order, order_ref=order_ref, column=column, arg_name=arg_name)

    def resolve_logic_map(self, *, ref: str) -> ResolvedLogicMap:
        return self.logic_maps.resolve(ref=ref)

    def resolve_plot_partition(self, *, partition: dict[str, Any] | Any | None) -> ResolvedPlotPartition:
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
        if collection_ref_raw is not None and (
            not isinstance(collection_ref_raw, str) or not collection_ref_raw.strip()
        ):
            raise ValueError("partition.collection_ref must be a non-empty string when provided")
        collection_ref = str(collection_ref_raw).strip() if isinstance(collection_ref_raw, str) else None
        match = partition.get("match", "exact")
        valid_match = {"exact", "contains", "startswith", "endswith", "regex"}
        if not isinstance(match, str) or match not in valid_match:
            raise ValueError(f"partition.match must be one of {sorted(valid_match)}")
        if collection_ref is None:
            return ResolvedPlotPartition(group_by=group_by, collection_items=None, match=match)
        collection = self.collections.resolve(ref=collection_ref)
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


@dataclass(frozen=True)
class ResourceEntry:
    kind: ResourceKind
    path: Path


@dataclass(frozen=True)
class ResourceCatalog:
    by_id: dict[str, ResourceEntry] = field(default_factory=dict)

    def get(self, resource_id: str) -> ResourceEntry | None:
        return self.by_id.get(resource_id)

    def require(self, resource_id: str) -> ResourceEntry:
        resource = self.get(resource_id)
        if resource is None:
            options = ", ".join(sorted(self.by_id)) if self.by_id else "—"
            raise ValueError(f"Unknown resource '{resource_id}'. Declare it under resources. (available: {options})")
        return resource

    def require_file(self, resource_id: str) -> ResourceEntry:
        resource = self.require(resource_id)
        if resource.kind != "file":
            raise ValueError(f"Resource '{resource_id}' has kind '{resource.kind}', expected file")
        return resource


@dataclass(frozen=True)
class OutputLayout:
    outputs_dir: Path
    plots_subdir: str
    exports_subdir: str
    notebooks_subdir: str

    def subdir_path(self, key: Literal["plots", "exports", "notebooks"]) -> Path:
        raw = {
            "plots": self.plots_subdir,
            "exports": self.exports_subdir,
            "notebooks": self.notebooks_subdir,
        }[key]
        return self.outputs_dir if raw in ("", ".", "./") else self.outputs_dir / raw


@dataclass(frozen=True)
class ExperimentSemantics:
    protocol: ProtocolBinding
    annotations: AnnotationSemantics
    resources: ResourceCatalog
    layout: OutputLayout
