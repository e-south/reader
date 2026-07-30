from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from reader_workbench.protocols.model import ProtocolBinding, ProtocolSemanticProgram

ResourceKind = Literal["file", "record"]
ReplicateKind = Literal["biological", "technical", "mixed", "unknown", "not_applicable"]


@dataclass(frozen=True)
class ExperimentEvidence:
    data_class: str
    data_class_reason: str
    replicate_kind: ReplicateKind
    replicate_identity_field: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data_class, str):
            raise TypeError("data_class must be a string")
        if not isinstance(self.data_class_reason, str):
            raise TypeError("data_class_reason must be a string")
        if not isinstance(self.replicate_kind, str):
            raise TypeError("replicate_kind must be a string")
        if self.replicate_identity_field is not None and not isinstance(self.replicate_identity_field, str):
            raise TypeError("replicate_identity_field must be a string when provided")
        data_class = self.data_class.strip()
        data_class_reason = self.data_class_reason.strip()
        replicate_identity_field = (
            self.replicate_identity_field.strip() if self.replicate_identity_field is not None else None
        )
        if not data_class:
            raise ValueError("data_class must be a non-empty string")
        if not data_class_reason:
            raise ValueError("data_class_reason must be a non-empty string")
        if self.replicate_kind not in {"biological", "technical", "mixed", "unknown", "not_applicable"}:
            raise ValueError(f"unsupported replicate_kind: {self.replicate_kind!r}")
        if self.replicate_identity_field is not None and not replicate_identity_field:
            raise ValueError("replicate_identity_field must be a non-empty string when provided")
        if self.replicate_kind == "not_applicable" and self.replicate_identity_field is not None:
            raise ValueError("replicate_identity_field cannot be set when replicate_kind is not_applicable")
        object.__setattr__(self, "data_class", data_class)
        object.__setattr__(self, "data_class_reason", data_class_reason)
        object.__setattr__(self, "replicate_identity_field", replicate_identity_field)

    def to_payload(self) -> dict[str, str | None]:
        return {
            "data_class": self.data_class,
            "data_class_reason": self.data_class_reason,
            "replicate_kind": self.replicate_kind,
            "replicate_identity_field": self.replicate_identity_field,
        }


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
class OrderedStateSpaceSpec:
    column: str
    state_order: tuple[str, ...]
    source_values: dict[str, str]
    case_sensitive: bool = True


@dataclass(frozen=True)
class ResolvedOrderedStateSpace:
    ref: str
    column: str
    state_ids: tuple[str, ...]
    source_values: dict[str, str]
    case_sensitive: bool


@dataclass(frozen=True)
class OrderedStateSpaces:
    by_id: dict[str, OrderedStateSpaceSpec] = field(default_factory=dict)

    def resolve(self, *, ref: str) -> ResolvedOrderedStateSpace:
        state_map_ref = str(ref).strip()
        if not state_map_ref:
            raise ValueError("state_map_ref must be a non-empty string")
        spec = self.by_id.get(state_map_ref)
        if spec is None:
            options = ", ".join(sorted(self.by_id)) if self.by_id else "—"
            raise ValueError(
                f"Unknown state_map_ref '{state_map_ref}'. "
                f"Define it under annotations.ordered_state_spaces.{state_map_ref}. (available: {options})"
            )
        if not isinstance(spec.column, str) or not spec.column.strip():
            raise ValueError(f"annotations.ordered_state_spaces.{state_map_ref}.column must be a non-empty string")
        state_ids = tuple(str(state_id).strip() for state_id in spec.state_order)
        if not state_ids:
            raise ValueError(f"annotations.ordered_state_spaces.{state_map_ref}.state_order must not be empty")
        if any(not state_id for state_id in state_ids):
            raise ValueError(f"annotations.ordered_state_spaces.{state_map_ref} state ids must be non-empty")
        if len(set(state_ids)) != len(state_ids):
            raise ValueError(f"annotations.ordered_state_spaces.{state_map_ref} state ids must be unique")
        if set(spec.source_values) != set(state_ids):
            raise ValueError(
                f"annotations.ordered_state_spaces.{state_map_ref}.values must have exactly the ids in state_order"
            )
        source_values = tuple(str(spec.source_values[state_id]) for state_id in state_ids)
        if any(not source_value.strip() for source_value in source_values):
            raise ValueError(f"annotations.ordered_state_spaces.{state_map_ref} source values must be non-empty")
        comparison_values = (
            source_values if spec.case_sensitive else tuple(value.strip().casefold() for value in source_values)
        )
        if len(set(comparison_values)) != len(comparison_values):
            sensitivity = "true" if spec.case_sensitive else "false"
            raise ValueError(
                f"annotations.ordered_state_spaces.{state_map_ref} source values must be unique "
                f"under case_sensitive={sensitivity}"
            )
        return ResolvedOrderedStateSpace(
            ref=state_map_ref,
            column=spec.column,
            state_ids=state_ids,
            source_values=dict(zip(state_ids, source_values, strict=True)),
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
    ordered_state_spaces: OrderedStateSpaces = field(default_factory=OrderedStateSpaces)

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

    def resolve_ordered_state_space(self, *, ref: str) -> ResolvedOrderedStateSpace:
        return self.ordered_state_spaces.resolve(ref=ref)

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
class FileResourceEntry:
    kind: Literal["file"]
    path: Path


@dataclass(frozen=True)
class RecordResourceEntry:
    kind: Literal["record"]
    experiment_id: str
    record_id: str
    experiment_root: Path
    outputs_dir: Path


ResourceEntry = FileResourceEntry | RecordResourceEntry


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

    def require_file(self, resource_id: str) -> FileResourceEntry:
        resource = self.require(resource_id)
        if resource.kind != "file":
            raise ValueError(f"Resource '{resource_id}' has kind '{resource.kind}', expected file")
        return resource

    def require_record(self, resource_id: str) -> RecordResourceEntry:
        resource = self.require(resource_id)
        if resource.kind != "record":
            raise ValueError(f"Resource '{resource_id}' has kind '{resource.kind}', expected record")
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
    protocol_program: ProtocolSemanticProgram
    evidence: ExperimentEvidence | None = None

    def __post_init__(self) -> None:
        if self.protocol_program.protocol != self.protocol.id:
            raise ValueError(
                "ExperimentSemantics.protocol_program must target the bound protocol "
                f"{self.protocol.id!r}, got {self.protocol_program.protocol!r}."
            )
