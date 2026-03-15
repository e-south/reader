from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from reader.core.errors import ConfigError

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
    if schema != "reader/v3":
        raise ConfigError(f"Config schema must be 'reader/v3'. This repo only supports reader/v3 (found {schema!r}).")

    removed_top_level_keys = {
        "steps",
        "overrides",
        "collections",
        "deliverables",
        "deliverable_presets",
        "deliverable_overrides",
        "notebook",
        "data",
        "semantics",
    }
    removed_experiment_keys = {"name", "outputs", "plots_dir", "palette"}
    illegal = sorted(key for key in removed_top_level_keys if key in data)
    if "experiment" in data and isinstance(data["experiment"], dict):
        illegal_exp = sorted(key for key in removed_experiment_keys if key in data["experiment"])
    else:
        illegal_exp = []
    if illegal or illegal_exp:
        parts = []
        if illegal:
            parts.append(f"top-level keys: {illegal}")
        if illegal_exp:
            parts.append(f"experiment keys: {illegal_exp}")
        raise ConfigError(
            "Unsupported legacy/removed config keys are not supported in reader/v3. Remove/replace: " + "; ".join(parts)
        )

    if "pipeline" not in data or not isinstance(data["pipeline"], dict):
        raise ConfigError("pipeline must be a mapping and include steps")
    if "steps" not in data["pipeline"]:
        raise ConfigError("pipeline.steps is required (use an empty list if there are no pipeline steps).")

    root = path.parent.resolve()

    data.setdefault("experiment", {})
    if not isinstance(data["experiment"], dict):
        raise ConfigError("experiment must be a mapping when provided")
    data.setdefault("paths", {})
    if not isinstance(data["paths"], dict):
        raise ConfigError("paths must be a mapping")
    data.setdefault("plotting", {})
    if not isinstance(data["plotting"], dict):
        raise ConfigError("plotting must be a mapping")
    data.setdefault("resources", {})
    if not isinstance(data["resources"], dict):
        raise ConfigError("resources must be a mapping of resource_id -> {kind, path}")
    data.setdefault("assay", {})
    if not isinstance(data["assay"], dict):
        raise ConfigError("assay must be a mapping")
    data.setdefault("plots", {})
    if not isinstance(data["plots"], dict):
        raise ConfigError("plots must be a mapping")
    data.setdefault("exports", {})
    if not isinstance(data["exports"], dict):
        raise ConfigError("exports must be a mapping")
    data.setdefault("notebooks", {})
    if not isinstance(data["notebooks"], dict):
        raise ConfigError("notebooks must be a mapping")
    if "steps" in data["plots"]:
        raise ConfigError("plots.steps is not supported in reader/v3. Use plots.specs.")
    if "steps" in data["exports"]:
        raise ConfigError("exports.steps is not supported in reader/v3. Use exports.specs.")
    if "steps" in data["notebooks"]:
        raise ConfigError("notebooks.steps is not supported in reader/v3. Use notebooks.specs.")
    notebook_removed = sorted(key for key in ("defaults", "overrides") if key in data["notebooks"])
    if notebook_removed:
        raise ConfigError(
            f"notebooks only supports specs in reader/v3. Remove notebooks.{', notebooks.'.join(notebook_removed)}."
        )

    for section in ("plots", "exports"):
        defaults = data[section].get("defaults", {}) or {}
        if not isinstance(defaults, dict):
            raise ConfigError(f"{section}.defaults must be a mapping")
        reads_default = defaults.get("reads", {}) or {}
        if not isinstance(reads_default, dict):
            raise ConfigError(f"{section}.defaults.reads must be a mapping")
        with_default = defaults.get("with", {}) or {}
        if not isinstance(with_default, dict):
            raise ConfigError(f"{section}.defaults.with must be a mapping")
        data[section]["defaults"] = {"reads": reads_default, "with": with_default}
        overrides = data[section].get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ConfigError(f"{section}.overrides must be a mapping of id -> overrides")
        data[section]["overrides"] = overrides

    outputs_raw = data["paths"].get("outputs", "./outputs")
    if not isinstance(outputs_raw, str) or not outputs_raw.strip():
        raise ConfigError("paths.outputs must be a non-empty string path")
    outputs_path = Path(outputs_raw).expanduser()
    if not outputs_path.is_absolute():
        outputs_path = (root / outputs_path).resolve()
    data["paths"]["outputs"] = str(outputs_path)

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
        resolved_subdir = (outputs_path / subdir).resolve()
        try:
            normalized_subdir = resolved_subdir.relative_to(outputs_path)
        except ValueError as exc:
            raise ConfigError(f"paths.{key} must stay under paths.outputs and may not escape via '..'.") from exc
        data["paths"][key] = str(normalized_subdir)

    palette_raw = data["plotting"].get("palette", None)
    if palette_raw is not None and (not isinstance(palette_raw, str) or not palette_raw.strip()):
        raise ConfigError("plotting.palette must be a non-empty string or null")

    resources_raw = data["resources"] or {}
    normalized_resources: dict[str, dict[str, str]] = {}
    for resource_id, resource in resources_raw.items():
        if not isinstance(resource, dict):
            raise ConfigError(f"resources.{resource_id} must be a mapping with kind/path")
        kind = resource.get("kind")
        path_raw = resource.get("path")
        if kind not in {"file", "directory"}:
            raise ConfigError(f"resources.{resource_id}.kind must be 'file' or 'directory'")
        if not isinstance(path_raw, str) or not path_raw.strip():
            raise ConfigError(f"resources.{resource_id}.path must be a non-empty string")
        path = Path(path_raw).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        normalized_resources[str(resource_id)] = {"kind": str(kind), "path": str(path)}
    data["resources"] = {"by_id": normalized_resources}

    assay_raw = data["assay"] or {}
    labels_raw = assay_raw.get("labels", {}) or {}
    if not isinstance(labels_raw, dict):
        raise ConfigError("assay.labels must be a mapping")
    normalized_labels: dict[str, dict[str, Any]] = {}
    for label_id, label_spec in labels_raw.items():
        if not isinstance(label_spec, dict):
            raise ConfigError(f"assay.labels.{label_id} must be a mapping")
        source = label_spec.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ConfigError(f"assay.labels.{label_id}.source must be a non-empty string")
        values = label_spec.get("values", {}) or {}
        if not isinstance(values, dict):
            raise ConfigError(f"assay.labels.{label_id}.values must be a mapping")
        output = label_spec.get("output")
        if output is not None and (not isinstance(output, str) or not output.strip()):
            raise ConfigError(f"assay.labels.{label_id}.output must be a non-empty string when provided")
        normalized_labels[str(label_id)] = {
            "source": source,
            "values": {str(k): str(v) for k, v in values.items()},
            "output": (str(output) if isinstance(output, str) else None),
        }

    orders_raw = assay_raw.get("orders", {}) or {}
    if not isinstance(orders_raw, dict):
        raise ConfigError("assay.orders must be a mapping")
    normalized_orders: dict[str, dict[str, Any]] = {}
    for order_id, order_spec in orders_raw.items():
        if not isinstance(order_spec, dict):
            raise ConfigError(f"assay.orders.{order_id} must be a mapping")
        column = order_spec.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ConfigError(f"assay.orders.{order_id}.column must be a non-empty string")
        values = order_spec.get("values", []) or []
        if not isinstance(values, list) or any(isinstance(item, (dict, list)) for item in values):
            raise ConfigError(f"assay.orders.{order_id}.values must be a flat list of scalar labels")
        if not values:
            raise ConfigError(f"assay.orders.{order_id}.values must not be empty")
        normalized_orders[str(order_id)] = {"column": column, "values": [str(item) for item in values]}

    collections_raw = assay_raw.get("collections", {}) or {}
    if not isinstance(collections_raw, dict):
        raise ConfigError("assay.collections must be a mapping")
    normalized_collections: dict[str, dict[str, Any]] = {}
    for collection_id, collection_spec in collections_raw.items():
        if not isinstance(collection_spec, dict):
            raise ConfigError(f"assay.collections.{collection_id} must be a mapping")
        column = collection_spec.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ConfigError(f"assay.collections.{collection_id}.column must be a non-empty string")
        items = collection_spec.get("items")
        if not isinstance(items, dict) or not items:
            raise ConfigError(f"assay.collections.{collection_id}.items must be a non-empty mapping")
        normalized_items: dict[str, list[str]] = {}
        for label, values in items.items():
            if not isinstance(values, list) or not values:
                raise ConfigError(f"assay.collections.{collection_id}.items.{label} must be a non-empty list")
            normalized_items[str(label)] = [str(item) for item in values]
        normalized_collections[str(collection_id)] = {
            "column": column,
            "items": normalized_items,
        }

    logic_maps_raw = assay_raw.get("logic_maps", {}) or {}
    if not isinstance(logic_maps_raw, dict):
        raise ConfigError("assay.logic_maps must be a mapping")
    normalized_logic_maps: dict[str, dict[str, Any]] = {}
    for logic_id, logic_spec in logic_maps_raw.items():
        if not isinstance(logic_spec, dict):
            raise ConfigError(f"assay.logic_maps.{logic_id} must be a mapping")
        column = logic_spec.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ConfigError(f"assay.logic_maps.{logic_id}.column must be a non-empty string")
        corners = logic_spec.get("corners")
        if not isinstance(corners, dict):
            raise ConfigError(f"assay.logic_maps.{logic_id}.corners must be a mapping")
        keys = set(corners)
        if keys != {"00", "10", "01", "11"}:
            raise ConfigError(
                f"assay.logic_maps.{logic_id}.corners must have exactly the keys ['00', '01', '10', '11']"
            )
        case_sensitive = logic_spec.get("case_sensitive", True)
        if not isinstance(case_sensitive, bool):
            raise ConfigError(f"assay.logic_maps.{logic_id}.case_sensitive must be true/false")
        normalized_logic_maps[str(logic_id)] = {
            "column": column,
            "corners": {str(k): str(v) for k, v in corners.items()},
            "case_sensitive": case_sensitive,
        }
    data["assay"] = {
        "labels": normalized_labels,
        "orders": normalized_orders,
        "collections": normalized_collections,
        "logic_maps": normalized_logic_maps,
    }

    runtime_raw = data["pipeline"].get("runtime", {}) or {}
    if not isinstance(runtime_raw, dict):
        raise ConfigError("pipeline.runtime must be a mapping")
    if "strict" in runtime_raw and not isinstance(runtime_raw["strict"], bool):
        raise ConfigError("pipeline.runtime.strict must be a boolean (true/false)")
    data["pipeline"]["runtime"] = runtime_raw

    pipeline_presets = _ensure_preset_list(data["pipeline"].get("presets", []) or [], section="pipeline")
    plots_presets = _ensure_preset_list(data["plots"].get("presets", []) or [], section="plots")
    exports_presets = _ensure_preset_list(data["exports"].get("presets", []) or [], section="exports")

    pipeline_steps = _ensure_step_list(data["pipeline"].get("steps", []) or [], section="pipeline", label="steps")
    plots_specs = _ensure_step_list(data["plots"].get("specs", []) or [], section="plots", label="specs")
    exports_specs = _ensure_step_list(data["exports"].get("specs", []) or [], section="exports", label="specs")
    notebooks_specs = _ensure_step_list(data["notebooks"].get("specs", []) or [], section="notebooks", label="specs")

    data["pipeline"]["presets"] = pipeline_presets
    data["plots"]["presets"] = plots_presets
    data["exports"]["presets"] = exports_presets
    data["pipeline"]["steps"] = pipeline_steps
    data["plots"]["specs"] = plots_specs
    data["exports"]["specs"] = exports_specs
    data["notebooks"]["specs"] = notebooks_specs

    experiment = data["experiment"]
    experiment_id = experiment.get("id")
    if experiment_id is None or (isinstance(experiment_id, str) and not experiment_id.strip()):
        experiment["id"] = root.name
    experiment_title = experiment.get("title")
    if experiment_title is None or (isinstance(experiment_title, str) and not experiment_title.strip()):
        experiment["title"] = experiment["id"]
    data["experiment"]["root"] = str(root)

    try:
        return cls.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, err.get('loc', [])))}: {err.get('msg')}" for err in exc.errors())
        raise ConfigError(f"Invalid config in {path}: {details}") from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _ensure_step_list(raw_steps: Any, *, section: str, label: str) -> list[dict[str, Any]]:
    if not isinstance(raw_steps, list):
        raise ConfigError(f"{section}.{label} must be a list")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_steps, 1):
        if not isinstance(entry, dict):
            raise ConfigError(f"{section}.{label} entry #{index} must be a mapping")
        if "preset" in entry:
            if section == "notebooks":
                raise ConfigError(
                    f"{section}.{label} does not support inline preset expansion; use a notebook template in the uses field instead."
                )
            raise ConfigError(
                f"{section}.{label} does not support inline preset expansion; use {section}.presets instead."
            )
        normalized.append(entry)
    return normalized


def _ensure_preset_list(raw_presets: Any, *, section: str) -> list[str | dict[str, Any]]:
    if not isinstance(raw_presets, list):
        raise ConfigError(f"{section}.presets must be a list")
    normalized: list[str | dict[str, Any]] = []
    for index, entry in enumerate(raw_presets, 1):
        if isinstance(entry, str):
            if not entry.strip():
                raise ConfigError(f"{section}.presets entry #{index} must be a non-empty string")
            normalized.append(entry)
            continue
        if not isinstance(entry, dict):
            raise ConfigError(f"{section}.presets entry #{index} must be a string or mapping")
        uses = entry.get("uses")
        if not isinstance(uses, str) or not uses.strip():
            raise ConfigError(f"{section}.presets entry #{index}: uses must be a non-empty string")
        with_block = entry.get("with", {}) or {}
        if not isinstance(with_block, dict):
            raise ConfigError(f"{section}.presets entry #{index}: with must be a mapping")
        normalized.append({"uses": uses, "with": with_block})
    return normalized
