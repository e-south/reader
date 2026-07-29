from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from reader.errors import ConfigError

from .model import ReaderSpec


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_reader_config_document(path: Path) -> dict[str, Any]:
    """Load a Reader config document with strict YAML and schema identity checks."""

    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config must be a mapping (YAML object) in {path}. Check for empty files or top-level lists."
        )

    schema = data.get("schema")
    if schema != "reader/v8":
        raise ConfigError(f"Config schema must be 'reader/v8'. This repo only supports reader/v8 (found {schema!r}).")
    return data


def load_reader_spec(path: Path, *, cls: type[ReaderSpec]) -> ReaderSpec:
    data = load_reader_config_document(path)

    removed_top_level_keys = {
        "steps",
        "overrides",
        "collections",
        "graph_patch",
        "deliverable_presets",
        "deliverable_overrides",
        "notebook",
        "data",
        "semantics",
        "assay",
        "pipeline",
        "plots",
        "exports",
        "notebooks",
    }
    removed_protocol_keys = {"with", "plugins", "parameters", "deliverables"}
    removed_experiment_keys = {"name", "outputs", "plots_dir", "palette"}
    illegal = sorted(key for key in removed_top_level_keys if key in data)
    illegal_protocol = []
    if "protocol" in data and isinstance(data["protocol"], dict):
        illegal_protocol = sorted(key for key in removed_protocol_keys if key in data["protocol"])
    illegal_exp = []
    if "experiment" in data and isinstance(data["experiment"], dict):
        illegal_exp = sorted(key for key in removed_experiment_keys if key in data["experiment"])
    if illegal or illegal_protocol or illegal_exp:
        parts = []
        if illegal:
            parts.append(f"top-level keys: {illegal}")
        if illegal_protocol:
            parts.append(f"protocol keys: {illegal_protocol}")
        if illegal_exp:
            parts.append(f"experiment keys: {illegal_exp}")
        raise ConfigError("reader/v8 rejects removed config keys. Remove or replace: " + "; ".join(parts))

    data.setdefault("experiment", {})
    if not isinstance(data["experiment"], dict):
        raise ConfigError("experiment must be a mapping when provided")
    experiment_id = data["experiment"].get("id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ConfigError("experiment.id is required and must be a non-empty string")
    data["experiment"].setdefault("title", experiment_id)

    protocol = data.get("protocol")
    if not isinstance(protocol, dict):
        raise ConfigError("protocol is required and must be a mapping with id/inputs/analysis/outputs")
    _ensure_only_keys(protocol, {"id", "inputs", "analysis", "outputs"}, where="protocol")
    protocol_id = protocol.get("id")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ConfigError("protocol.id must be a non-empty string")
    inputs = protocol.get("inputs", {}) or {}
    analysis = protocol.get("analysis", {}) or {}
    outputs = protocol.get("outputs", {}) or {}
    if not isinstance(inputs, dict):
        raise ConfigError("protocol.inputs must be a mapping")
    if not isinstance(analysis, dict):
        raise ConfigError("protocol.analysis must be a mapping")
    if not isinstance(outputs, dict):
        raise ConfigError("protocol.outputs must be a mapping")
    data["protocol"] = {
        "id": protocol_id.strip(),
        "inputs": dict(inputs),
        "analysis": dict(analysis),
        "outputs": _normalize_outputs(outputs),
    }

    data.setdefault("paths", {})
    if not isinstance(data["paths"], dict):
        raise ConfigError("paths must be a mapping")
    outputs_raw = data["paths"].get("outputs", "./outputs")
    if not isinstance(outputs_raw, str) or not outputs_raw.strip():
        raise ConfigError("paths.outputs must be a non-empty string path")
    data["paths"]["outputs"] = outputs_raw
    for key, value in (
        ("plots", data["paths"].get("plots", "plots")),
        ("exports", data["paths"].get("exports", "exports")),
        ("notebooks", data["paths"].get("notebooks", "notebooks")),
    ):
        if value is None:
            raise ConfigError(f"paths.{key} must be a string subdirectory (use '.' to flatten).")
        if not isinstance(value, str):
            raise ConfigError(f"paths.{key} must be a string subdirectory")
        subdir = Path(value)
        if subdir.is_absolute():
            raise ConfigError(f"paths.{key} must be relative to paths.outputs, not absolute.")
        normalized_subdir = (Path(".") / subdir).parts
        if ".." in normalized_subdir:
            raise ConfigError(f"paths.{key} must stay under paths.outputs and may not escape via '..'.")
        data["paths"][key] = value

    data.setdefault("plotting", {})
    if not isinstance(data["plotting"], dict):
        raise ConfigError("plotting must be a mapping")
    palette_raw = data["plotting"].get("palette", None)
    if palette_raw is not None and (not isinstance(palette_raw, str) or not palette_raw.strip()):
        raise ConfigError("plotting.palette must be a non-empty string or null")

    data.setdefault("resources", {})
    if not isinstance(data["resources"], dict):
        raise ConfigError("resources must be a mapping of resource_id -> {kind, path}")
    normalized_resources: dict[str, dict[str, str]] = {}
    for resource_id, resource in (data["resources"] or {}).items():
        if not isinstance(resource, dict):
            raise ConfigError(f"resources.{resource_id} must be a mapping with kind/path")
        kind = resource.get("kind")
        path_raw = resource.get("path")
        if kind != "file":
            raise ConfigError(f"resources.{resource_id}.kind must be 'file'")
        if not isinstance(path_raw, str) or not path_raw.strip():
            raise ConfigError(f"resources.{resource_id}.path must be a non-empty string")
        normalized_resources[str(resource_id)] = {"kind": str(kind), "path": str(path_raw)}
    data["resources"] = {"by_id": normalized_resources}

    data.setdefault("annotations", {})
    if not isinstance(data["annotations"], dict):
        raise ConfigError("annotations must be a mapping")
    data["annotations"] = _normalize_annotations(data["annotations"])

    try:
        return cls.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def _normalize_annotations(annotations_raw: dict[str, Any]) -> dict[str, Any]:
    _ensure_only_keys(
        annotations_raw,
        {"labels", "orders", "collections", "ordered_state_spaces"},
        where="annotations",
    )
    labels_raw = annotations_raw.get("labels", {}) or {}
    if not isinstance(labels_raw, dict):
        raise ConfigError("annotations.labels must be a mapping")
    normalized_labels: dict[str, dict[str, Any]] = {}
    for label_id, label_spec in labels_raw.items():
        if not isinstance(label_spec, dict):
            raise ConfigError(f"annotations.labels.{label_id} must be a mapping")
        _ensure_only_keys(label_spec, {"source", "values", "output"}, where=f"annotations.labels.{label_id}")
        source = label_spec.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ConfigError(f"annotations.labels.{label_id}.source must be a non-empty string")
        values = label_spec.get("values", {}) or {}
        if not isinstance(values, dict):
            raise ConfigError(f"annotations.labels.{label_id}.values must be a mapping")
        output = label_spec.get("output")
        if output is not None and (not isinstance(output, str) or not output.strip()):
            raise ConfigError(f"annotations.labels.{label_id}.output must be a non-empty string when provided")
        normalized_labels[str(label_id)] = {
            "source": source,
            "values": {str(k): str(v) for k, v in values.items()},
            "output": (str(output) if isinstance(output, str) else None),
        }

    orders_raw = annotations_raw.get("orders", {}) or {}
    if not isinstance(orders_raw, dict):
        raise ConfigError("annotations.orders must be a mapping")
    normalized_orders: dict[str, dict[str, Any]] = {}
    for order_id, order_spec in orders_raw.items():
        if not isinstance(order_spec, dict):
            raise ConfigError(f"annotations.orders.{order_id} must be a mapping")
        _ensure_only_keys(order_spec, {"column", "values"}, where=f"annotations.orders.{order_id}")
        column = order_spec.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ConfigError(f"annotations.orders.{order_id}.column must be a non-empty string")
        values = order_spec.get("values", []) or []
        if not isinstance(values, list) or any(isinstance(item, (dict, list)) for item in values):
            raise ConfigError(f"annotations.orders.{order_id}.values must be a flat list of scalar labels")
        if not values:
            raise ConfigError(f"annotations.orders.{order_id}.values must not be empty")
        normalized_orders[str(order_id)] = {"column": column, "values": [str(item) for item in values]}

    collections_raw = annotations_raw.get("collections", {}) or {}
    if not isinstance(collections_raw, dict):
        raise ConfigError("annotations.collections must be a mapping")
    normalized_collections: dict[str, dict[str, Any]] = {}
    for collection_id, collection_spec in collections_raw.items():
        if not isinstance(collection_spec, dict):
            raise ConfigError(f"annotations.collections.{collection_id} must be a mapping")
        _ensure_only_keys(collection_spec, {"column", "items"}, where=f"annotations.collections.{collection_id}")
        column = collection_spec.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ConfigError(f"annotations.collections.{collection_id}.column must be a non-empty string")
        items = collection_spec.get("items", {}) or {}
        if not isinstance(items, dict):
            raise ConfigError(f"annotations.collections.{collection_id}.items must be a mapping")
        invalid_items = sorted(item_key for item_key, item_values in items.items() if not isinstance(item_values, list))
        if invalid_items:
            raise ConfigError(
                f"annotations.collections.{collection_id}.items entries must be lists for keys: {invalid_items}"
            )
        normalized_collections[str(collection_id)] = {
            "column": column,
            "items": {str(item_key): [str(item) for item in item_values] for item_key, item_values in items.items()},
        }

    state_spaces_raw = annotations_raw.get("ordered_state_spaces", {}) or {}
    if not isinstance(state_spaces_raw, dict):
        raise ConfigError("annotations.ordered_state_spaces must be a mapping")
    normalized_state_spaces: dict[str, dict[str, Any]] = {}
    for space_id, space_spec in state_spaces_raw.items():
        if not isinstance(space_id, str) or not space_id or space_id != space_id.strip():
            raise ConfigError("annotations.ordered_state_spaces keys must be non-empty, already-trimmed strings")
        context = f"annotations.ordered_state_spaces.{space_id}"
        if not isinstance(space_spec, dict):
            raise ConfigError(f"{context} must be a mapping")
        _ensure_only_keys(space_spec, {"column", "state_order", "values", "case_sensitive"}, where=context)
        column = space_spec.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ConfigError(f"{context}.column must be a non-empty string")
        state_order = space_spec.get("state_order")
        if not isinstance(state_order, list) or not state_order:
            raise ConfigError(f"{context}.state_order must be a non-empty list")
        if any(
            not isinstance(state_id, str) or not state_id or state_id != state_id.strip() for state_id in state_order
        ):
            raise ConfigError(f"{context}.state_order must contain non-empty, already-trimmed strings")
        normalized_state_order = list(state_order)
        if len(set(normalized_state_order)) != len(normalized_state_order):
            raise ConfigError(f"{context} state ids must be unique")
        values = space_spec.get("values")
        if not isinstance(values, dict) or not values:
            raise ConfigError(f"{context}.values must be a non-empty mapping")
        if any(not isinstance(state_id, str) for state_id in values):
            raise ConfigError(f"{context}.values keys must be strings")
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ConfigError(f"{context}.values must map state ids to non-empty strings")
        normalized_values = dict(values)
        if set(normalized_values) != set(normalized_state_order):
            raise ConfigError(f"{context}.values must have exactly the ids declared by state_order")
        case_sensitive = space_spec.get("case_sensitive", True)
        if not isinstance(case_sensitive, bool):
            raise ConfigError(f"{context}.case_sensitive must be a boolean")
        comparison_values = [
            normalized_values[state_id] if case_sensitive else normalized_values[state_id].strip().casefold()
            for state_id in normalized_state_order
        ]
        if len(set(comparison_values)) != len(comparison_values):
            sensitivity = "true" if case_sensitive else "false"
            raise ConfigError(f"{context} source values must be unique under case_sensitive={sensitivity}")
        normalized_state_spaces[space_id] = {
            "column": column.strip(),
            "state_order": normalized_state_order,
            "values": normalized_values,
            "case_sensitive": case_sensitive,
        }

    return {
        "labels": normalized_labels,
        "orders": normalized_orders,
        "collections": normalized_collections,
        "ordered_state_spaces": normalized_state_spaces,
    }


def _normalize_outputs(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    _ensure_only_keys(raw, {"notebook", "plots", "exports"}, where="protocol.outputs")
    notebook = raw.get("notebook", {}) or {}
    if not isinstance(notebook, dict):
        raise ConfigError("protocol.outputs.notebook must be a mapping")
    _ensure_only_keys(notebook, {"template"}, where="protocol.outputs.notebook")
    template = notebook.get("template")
    if template is not None and (not isinstance(template, str) or not template.strip()):
        raise ConfigError("protocol.outputs.notebook.template must be a non-empty string when provided")
    normalized["notebook"] = {"template": template}

    plots = raw.get("plots", {}) or {}
    if not isinstance(plots, dict):
        raise ConfigError("protocol.outputs.plots must be a mapping")
    _ensure_only_keys(plots, {"profile", "include", "exclude", "views"}, where="protocol.outputs.plots")
    plot_profile = plots.get("profile")
    if plot_profile is not None and (not isinstance(plot_profile, str) or not plot_profile.strip()):
        raise ConfigError("protocol.outputs.plots.profile must be a non-empty string when provided")
    plot_include = plots.get("include", []) or []
    plot_exclude = plots.get("exclude", []) or []
    plot_views = plots.get("views", {}) or {}
    if not isinstance(plot_include, list) or not all(isinstance(item, str) and item.strip() for item in plot_include):
        raise ConfigError("protocol.outputs.plots.include must be a list of non-empty plot ids")
    if not isinstance(plot_exclude, list) or not all(isinstance(item, str) and item.strip() for item in plot_exclude):
        raise ConfigError("protocol.outputs.plots.exclude must be a list of non-empty plot ids")
    if not isinstance(plot_views, dict):
        raise ConfigError("protocol.outputs.plots.views must be a mapping of plot id -> view config")
    normalized["plots"] = {
        "profile": plot_profile,
        "include": [str(item) for item in plot_include],
        "exclude": [str(item) for item in plot_exclude],
        "views": {
            str(plot_id): _normalize_mapping(settings_block, where="protocol.outputs.plots.views")
            for plot_id, settings_block in plot_views.items()
        },
    }

    exports = raw.get("exports", {}) or {}
    if not isinstance(exports, dict):
        raise ConfigError("protocol.outputs.exports must be a mapping")
    _ensure_only_keys(exports, {"include", "exclude", "artifacts"}, where="protocol.outputs.exports")
    export_include = exports.get("include", []) or []
    export_exclude = exports.get("exclude", []) or []
    export_artifacts = exports.get("artifacts", {}) or {}
    if not isinstance(export_include, list) or not all(
        isinstance(item, str) and item.strip() for item in export_include
    ):
        raise ConfigError("protocol.outputs.exports.include must be a list of non-empty artifact ids")
    if not isinstance(export_exclude, list) or not all(
        isinstance(item, str) and item.strip() for item in export_exclude
    ):
        raise ConfigError("protocol.outputs.exports.exclude must be a list of non-empty artifact ids")
    if not isinstance(export_artifacts, dict):
        raise ConfigError("protocol.outputs.exports.artifacts must be a mapping of artifact id -> config")
    normalized["exports"] = {
        "include": [str(item) for item in export_include],
        "exclude": [str(item) for item in export_exclude],
        "artifacts": {
            str(artifact_id): _normalize_mapping(settings_block, where="protocol.outputs.exports.artifacts")
            for artifact_id, settings_block in export_artifacts.items()
        },
    }
    return normalized


def _normalize_mapping(raw: Any, *, where: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} entries must be mappings")
    return dict(raw)


def _ensure_only_keys(raw: dict[str, Any], allowed: set[str], *, where: str) -> None:
    unknown = sorted(key for key in raw if key not in allowed)
    if unknown:
        options = ", ".join(sorted(allowed)) or "—"
        raise ConfigError(f"{where} has unknown keys {unknown}. Allowed keys: {options}")
