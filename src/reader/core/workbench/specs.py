from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reader.core.config import ReaderSpec, StepSpec
from reader.core.errors import ConfigError
from reader.core.notebooks import resolve_notebook_template_descriptor
from reader.core.presets import resolve_preset

from .ontology import WorkbenchSpecKind, WorkbenchSpecSemantics, get_workbench_spec_semantics


@dataclass(frozen=True)
class WorkbenchSpec:
    kind: WorkbenchSpecKind
    id: str
    uses: str
    reads: dict[str, str] = field(default_factory=dict)
    with_: dict[str, Any] = field(default_factory=dict)
    writes: dict[str, str] = field(default_factory=dict)
    preset_meta: dict[str, Any] | None = None

    @property
    def semantics(self) -> WorkbenchSpecSemantics:
        return get_workbench_spec_semantics(self.kind)

    @property
    def uses_category(self) -> str:
        return self.uses.split("/", 1)[0] if "/" in self.uses else self.uses

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "uses": self.uses,
            "with": dict(self.with_ or {}),
        }
        if self.kind != "notebook":
            payload["reads"] = dict(self.reads or {})
            payload["writes"] = dict(self.writes or {})
        return payload


@dataclass(frozen=True)
class Workbench:
    pipeline: tuple[WorkbenchSpec, ...] = ()
    plots: tuple[WorkbenchSpec, ...] = ()
    exports: tuple[WorkbenchSpec, ...] = ()
    notebooks: tuple[WorkbenchSpec, ...] = ()

    def by_kind(self, kind: WorkbenchSpecKind) -> tuple[WorkbenchSpec, ...]:
        if kind == "pipeline":
            return self.pipeline
        if kind == "plot":
            return self.plots
        if kind == "export":
            return self.exports
        return self.notebooks

    def all_specs(self) -> tuple[WorkbenchSpec, ...]:
        return self.pipeline + self.plots + self.exports + self.notebooks

    def uses_categories(self) -> set[str]:
        return {item.uses_category for item in self.all_specs()}

    def plugin_specs(self) -> tuple[WorkbenchSpec, ...]:
        return tuple(item for item in self.all_specs() if item.semantics.plugin_backed)

    def plugin_categories(self) -> set[str]:
        return {item.uses_category for item in self.plugin_specs()}

    def counts(self) -> dict[WorkbenchSpecKind, int]:
        return {
            "pipeline": len(self.pipeline),
            "plot": len(self.plots),
            "export": len(self.exports),
            "notebook": len(self.notebooks),
        }


def ensure_unique_workbench_ids(*collections: Sequence[WorkbenchSpec]) -> None:
    ids: list[str] = [item.id for collection in collections for item in collection if item.id]
    dupes = sorted(item_id for item_id, count in Counter(ids).items() if count > 1)
    if dupes:
        raise ConfigError(f"Duplicate step/spec id(s) across pipeline/plots/exports/notebooks: {dupes}")


def resolve_workbench(spec: ReaderSpec) -> Workbench:
    pipeline = tuple(
        _resolve_specs(
            spec,
            kind="pipeline",
            presets=spec.pipeline.presets,
            defaults={"reads": {}, "with": {}},
            overrides=spec.pipeline.overrides,
            specs=spec.pipeline.steps,
        )
    )
    plots = tuple(
        _resolve_specs(
            spec,
            kind="plot",
            presets=spec.plots.presets,
            defaults={"reads": spec.plots.defaults.reads, "with": spec.plots.defaults.with_},
            overrides=spec.plots.overrides,
            specs=spec.plots.specs,
        )
    )
    exports = tuple(
        _resolve_specs(
            spec,
            kind="export",
            presets=spec.exports.presets,
            defaults={"reads": spec.exports.defaults.reads, "with": spec.exports.defaults.with_},
            overrides=spec.exports.overrides,
            specs=spec.exports.specs,
        )
    )
    notebooks = tuple(_resolve_notebook_specs(spec))
    ensure_unique_workbench_ids(pipeline, plots, exports, notebooks)
    return Workbench(pipeline=pipeline, plots=plots, exports=exports, notebooks=notebooks)


def materialize_workbench(spec: ReaderSpec) -> dict[str, list[dict[str, Any]]]:
    workbench = resolve_workbench(spec)
    return {
        "pipeline": [item.to_dict() for item in workbench.pipeline],
        "plots": [item.to_dict() for item in workbench.plots],
        "exports": [item.to_dict() for item in workbench.exports],
        "notebooks": [item.to_dict() for item in workbench.notebooks],
    }


def select_workbench_specs(
    specs: Sequence[WorkbenchSpec],
    *,
    only: Sequence[str],
    exclude: Sequence[str],
    kind_label: str,
) -> list[WorkbenchSpec]:
    ids = [item.id for item in specs]
    available = set(ids)
    if only:
        only_ids = set(only)
        missing = sorted(only_ids - available)
        if missing:
            raise ConfigError(f"Unknown {kind_label} id(s): {missing}.")
        selected = [item for item in specs if item.id in only_ids]
    else:
        selected = list(specs)
    if exclude:
        exclude_ids = set(exclude)
        missing = sorted(exclude_ids - available)
        if missing:
            raise ConfigError(f"Unknown {kind_label} id(s): {missing}.")
        selected = [item for item in selected if item.id not in exclude_ids]
    return selected


def _resolve_specs(
    spec: ReaderSpec,
    *,
    kind: WorkbenchSpecKind,
    presets: list[Any],
    defaults: dict[str, Any],
    overrides: dict[str, Any],
    specs: list[StepSpec],
) -> list[WorkbenchSpec]:
    semantics = get_workbench_spec_semantics(kind)
    root = Path(spec.experiment.root or ".").resolve()
    resources = dict(spec.resources.by_id or {})
    raw_steps: list[dict[str, Any]] = []

    for preset in presets or []:
        preset_name, preset_with = _normalize_preset_call(preset)
        expanded = resolve_preset(preset_name, with_args=preset_with)
        if not isinstance(expanded, list):
            raise ConfigError(f"{semantics.section}.presets '{preset_name}' did not resolve to a list of steps")
        for entry in expanded:
            if not isinstance(entry, dict):
                raise ConfigError(f"{semantics.section}.presets '{preset_name}' contains a non-mapping entry")
            enriched = dict(entry)
            enriched["preset_meta"] = {"preset": preset_name, "with": dict(preset_with)}
            raw_steps.append(enriched)

    for entry in specs or []:
        if isinstance(entry, StepSpec):
            raw_steps.append(entry.model_dump(by_alias=True))
        elif isinstance(entry, dict):
            raw_steps.append(entry)
        else:
            raise ConfigError(f"{semantics.section}.specs entry must be a mapping (got {type(entry).__name__})")

    if not raw_steps:
        return []

    default_reads = defaults.get("reads") or {}
    default_with = defaults.get("with") or {}
    if not isinstance(default_reads, dict):
        raise ConfigError(f"{semantics.section}.defaults.reads must be a mapping")
    if not isinstance(default_with, dict):
        raise ConfigError(f"{semantics.section}.defaults.with must be a mapping")

    finalized: list[dict[str, Any]] = []
    for step in raw_steps:
        step_id = step.get("id")
        if not step_id or not isinstance(step_id, str):
            raise ConfigError(f"Every {semantics.section} spec must include an id.")
        uses = step.get("uses")
        if not uses or not isinstance(uses, str):
            raise ConfigError(f"{semantics.section} {step_id}: uses must be a non-empty string")
        if "/" not in uses:
            raise ConfigError(f"{semantics.section} {step_id}: uses must be 'category/key'")
        category = uses.split("/", 1)[0]
        expected_category = semantics.uses_category
        if kind == "pipeline" and category in {"plot", "export", "notebook"}:
            raise ConfigError(f"pipeline {step_id}: plot/export/notebook plugins are not allowed in pipeline.")
        if expected_category is not None and category != expected_category:
            raise ConfigError(f"{semantics.section} {step_id}: uses must be {expected_category}/*")

        reads = step.get("reads") or {}
        if not isinstance(reads, dict):
            raise ConfigError(f"{semantics.section} {step_id}: reads must be a mapping")
        with_block = step.get("with") or {}
        if not isinstance(with_block, dict):
            raise ConfigError(f"{semantics.section} {step_id}: with must be a mapping")
        writes = step.get("writes") or {}
        if not isinstance(writes, dict):
            raise ConfigError(f"{semantics.section} {step_id}: writes must be a mapping")

        finalized.append(
            {
                "id": step_id,
                "uses": uses,
                "reads": {**default_reads, **reads},
                "with": {**default_with, **with_block},
                "writes": writes,
                "preset_meta": step.get("preset_meta") if isinstance(step.get("preset_meta"), dict) else None,
            }
        )

    if overrides:
        if not isinstance(overrides, dict):
            raise ConfigError(f"{semantics.section}.overrides must be a mapping of id -> overrides")
        ids = {item["id"] for item in finalized}
        unknown = sorted(set(overrides) - ids)
        if unknown:
            raise ConfigError(
                f"{semantics.section}.overrides reference unknown id(s): {unknown}. "
                "Check preset-expanded ids or remove stale overrides."
            )
        for index, step in enumerate(finalized):
            step_id = step["id"]
            if step_id not in overrides:
                continue
            merged = _deep_merge(step, overrides[step_id])
            if merged.get("id") != step_id:
                raise ConfigError(f"{semantics.section}.overrides for '{step_id}' cannot change the id.")
            finalized[index] = merged

    seen: set[str] = set()
    resolved: list[WorkbenchSpec] = []
    for step in finalized:
        step_id = step["id"]
        if step_id in seen:
            raise ConfigError(f"{semantics.section} contains duplicate spec id(s): {step_id}")
        seen.add(step_id)
        reads = step.get("reads") or {}
        writes = step.get("writes") or {}
        with_block = step.get("with") or {}
        if not isinstance(reads, dict):
            raise ConfigError(f"{semantics.section} {step_id}: reads must be a mapping")
        if not isinstance(writes, dict):
            raise ConfigError(f"{semantics.section} {step_id}: writes must be a mapping")
        if not isinstance(with_block, dict):
            raise ConfigError(f"{semantics.section} {step_id}: with must be a mapping")
        resolved.append(
            WorkbenchSpec(
                kind=kind,
                id=step_id,
                uses=step["uses"],
                reads=_normalize_external_reads(
                    reads,
                    root=root,
                    section=semantics.section,
                    step_id=step_id,
                    resources=resources,
                ),
                with_=dict(with_block),
                writes=dict(writes),
                preset_meta=step.get("preset_meta"),
            )
        )
    return resolved


def _resolve_notebook_specs(spec: ReaderSpec) -> list[WorkbenchSpec]:
    semantics = get_workbench_spec_semantics("notebook")
    default_with = spec.notebooks.defaults.with_ or {}
    if not isinstance(default_with, dict):
        raise ConfigError(f"{semantics.section}.defaults.with must be a mapping")

    raw_specs = [entry.model_dump(by_alias=True) for entry in (spec.notebooks.specs or [])]
    if not raw_specs:
        return []

    overrides = spec.notebooks.overrides or {}
    if not isinstance(overrides, dict):
        raise ConfigError(f"{semantics.section}.overrides must be a mapping of id -> overrides")

    ids = {entry.get("id") for entry in raw_specs}
    unknown = sorted(set(overrides) - ids)
    if unknown:
        raise ConfigError(
            f"{semantics.section}.overrides reference unknown id(s): {unknown}. "
            "Check configured notebook ids or remove stale overrides."
        )

    finalized: list[dict[str, Any]] = []
    for entry in raw_specs:
        step_id = entry.get("id")
        if not step_id or not isinstance(step_id, str):
            raise ConfigError(f"Every {semantics.section} spec must include an id.")
        if step_id in overrides:
            merged = _deep_merge(entry, overrides[step_id])
            if merged.get("id") != step_id:
                raise ConfigError(f"{semantics.section}.overrides for '{step_id}' cannot change the id.")
            entry = merged
        uses = entry.get("uses")
        if not uses or not isinstance(uses, str):
            raise ConfigError(f"{semantics.section} {step_id}: uses must be a non-empty string")
        if "/" not in uses:
            raise ConfigError(f"{semantics.section} {step_id}: uses must be 'category/key'")
        if semantics.uses_category is not None and uses.split("/", 1)[0] != semantics.uses_category:
            raise ConfigError(f"{semantics.section} {step_id}: uses must be {semantics.uses_category}/*")
        descriptor = resolve_notebook_template_descriptor(uses)
        uses = descriptor.uses
        with_block = entry.get("with") or {}
        if not isinstance(with_block, dict):
            raise ConfigError(f"{semantics.section} {step_id}: with must be a mapping")
        finalized.append({"id": step_id, "uses": uses, "with": {**default_with, **with_block}})

    seen: set[str] = set()
    resolved: list[WorkbenchSpec] = []
    for entry in finalized:
        step_id = entry["id"]
        if step_id in seen:
            raise ConfigError(f"{semantics.section} contains duplicate spec id(s): {step_id}")
        seen.add(step_id)
        resolved.append(
            WorkbenchSpec(
                kind="notebook",
                id=step_id,
                uses=entry["uses"],
                with_=dict(entry.get("with") or {}),
            )
        )
    return resolved


def _normalize_external_reads(
    reads: dict[str, Any],
    *,
    root: Path,
    section: str,
    step_id: str,
    resources: dict[str, Any],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in reads.items():
        if isinstance(value, str) and value.startswith("file:"):
            raw = value.split("file:", 1)[1].strip()
            if not raw:
                raise ConfigError(f"{section} {step_id}: reads '{key}' uses an empty file: path.")
            path = Path(raw).expanduser()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
            normalized[key] = f"file:{path}"
            continue
        if isinstance(value, str) and value.startswith("resource:"):
            resource_id = value.split("resource:", 1)[1].strip()
            if not resource_id:
                raise ConfigError(f"{section} {step_id}: reads '{key}' uses an empty resource: ref.")
            resource = resources.get(resource_id)
            if resource is None:
                raise ConfigError(f"{section} {step_id}: reads '{key}' references unknown resource '{resource_id}'.")
            if hasattr(resource, "model_dump"):
                resource = resource.model_dump()
            kind = resource.get("kind")
            if kind != "file":
                raise ConfigError(
                    f"{section} {step_id}: reads '{key}' references resource '{resource_id}', "
                    f"but kind '{kind}' is not readable as a file input."
                )
            normalized[key] = f"file:{resource['path']}"
            continue
        normalized[key] = value
    return normalized


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _normalize_preset_call(raw: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(raw, str):
        return raw, {}
    if hasattr(raw, "uses") and hasattr(raw, "with_"):
        return str(raw.uses), dict(raw.with_ or {})
    if isinstance(raw, dict):
        uses = raw.get("uses")
        with_block = raw.get("with", {}) or {}
        if not isinstance(uses, str) or not uses.strip():
            raise ConfigError("Preset call uses must be a non-empty string.")
        if not isinstance(with_block, dict):
            raise ConfigError(f"Preset call for {uses!r}: with must be a mapping.")
        return uses, dict(with_block)
    raise ConfigError(f"Unsupported preset call type: {type(raw).__name__}")
