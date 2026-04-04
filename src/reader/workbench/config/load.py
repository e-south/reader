from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from reader.errors import ConfigError

from .model import ReaderSpec


def load_reader_spec(path: Path, *, cls: type[ReaderSpec]) -> ReaderSpec:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config must be a mapping (YAML object) in {path}. Check for empty files or top-level lists."
        )

    schema = data.get("schema")
    if schema != "reader/v7":
        raise ConfigError(f"Config schema must be 'reader/v7'. This repo only supports reader/v7 (found {schema!r}).")

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
        raise ConfigError(
            "Unsupported legacy/removed config keys are not supported in reader/v7. Remove/replace: " + "; ".join(parts)
        )

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
        if kind not in {"file", "directory"}:
            raise ConfigError(f"resources.{resource_id}.kind must be 'file' or 'directory'")
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
    _ensure_only_keys(annotations_raw, {"labels", "orders", "collections", "logic_maps"}, where="annotations")
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

    logic_maps_raw = annotations_raw.get("logic_maps", {}) or {}
    if not isinstance(logic_maps_raw, dict):
        raise ConfigError("annotations.logic_maps must be a mapping")
    normalized_logic_maps: dict[str, dict[str, Any]] = {}
    for logic_id, logic_spec in logic_maps_raw.items():
        if not isinstance(logic_spec, dict):
            raise ConfigError(f"annotations.logic_maps.{logic_id} must be a mapping")
        _ensure_only_keys(
            logic_spec,
            {"column", "corners", "case_sensitive"},
            where=f"annotations.logic_maps.{logic_id}",
        )
        column = logic_spec.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ConfigError(f"annotations.logic_maps.{logic_id}.column must be a non-empty string")
        corners = logic_spec.get("corners")
        if not isinstance(corners, dict) or not corners:
            raise ConfigError(f"annotations.logic_maps.{logic_id}.corners must be a non-empty mapping")
        normalized_logic_maps[str(logic_id)] = {
            "column": column,
            "corners": {str(key): str(value) for key, value in corners.items()},
            "case_sensitive": bool(logic_spec.get("case_sensitive", True)),
        }

    return {
        "labels": normalized_labels,
        "orders": normalized_orders,
        "collections": normalized_collections,
        "logic_maps": normalized_logic_maps,
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
