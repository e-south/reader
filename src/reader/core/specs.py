"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/specs.py

Resolve plot/export specs from presets + config, applying defaults and overrides.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reader.core.config_model import ReaderSpec, StepSpec
from reader.core.errors import ConfigError
from reader.core.presets import resolve_preset


@dataclass
class PlotSpec:
    id: str
    uses: str
    reads: dict[str, str] = field(default_factory=dict)
    with_: dict[str, Any] = field(default_factory=dict)
    writes: dict[str, str] = field(default_factory=dict)
    preset_meta: dict[str, Any] | None = None


@dataclass
class ExportSpec:
    id: str
    uses: str
    reads: dict[str, str] = field(default_factory=dict)
    with_: dict[str, Any] = field(default_factory=dict)
    writes: dict[str, str] = field(default_factory=dict)
    preset_meta: dict[str, Any] | None = None


def ensure_unique_spec_ids(
    pipeline_steps: list[Any], plot_specs: list[PlotSpec], export_specs: list[ExportSpec]
) -> None:
    ids: list[str] = []
    for step in pipeline_steps + plot_specs + export_specs:
        step_id = getattr(step, "id", None)
        if step_id:
            ids.append(step_id)
    dupes = sorted({s for s in ids if ids.count(s) > 1})
    if dupes:
        raise ConfigError(f"Duplicate step/spec id(s) across pipeline/plots/exports: {dupes}")


def resolve_plot_specs(spec: ReaderSpec) -> list[PlotSpec]:
    return _resolve_specs(
        spec,
        section="plots",
        presets=spec.plots.presets,
        defaults={"reads": spec.plots.defaults.reads, "with": spec.plots.defaults.with_},
        overrides=spec.plots.overrides,
        specs=spec.plots.specs,
        spec_cls=PlotSpec,
    )


def resolve_export_specs(spec: ReaderSpec) -> list[ExportSpec]:
    return _resolve_specs(
        spec,
        section="exports",
        presets=spec.exports.presets,
        defaults={"reads": spec.exports.defaults.reads, "with": spec.exports.defaults.with_},
        overrides=spec.exports.overrides,
        specs=spec.exports.specs,
        spec_cls=ExportSpec,
    )


def materialize_specs(spec: ReaderSpec) -> dict[str, list[dict[str, Any]]]:
    def _to_dict(item: PlotSpec | ExportSpec) -> dict[str, Any]:
        return {
            "id": item.id,
            "uses": item.uses,
            "reads": dict(item.reads or {}),
            "with": dict(item.with_ or {}),
            "writes": dict(item.writes or {}),
        }

    return {
        "plots": [_to_dict(s) for s in resolve_plot_specs(spec)],
        "exports": [_to_dict(s) for s in resolve_export_specs(spec)],
    }


def _resolve_specs(
    spec: ReaderSpec,
    *,
    section: str,
    presets: list[str],
    defaults: dict[str, Any],
    overrides: dict[str, Any],
    specs: list[StepSpec],
    spec_cls: type[PlotSpec] | type[ExportSpec],
) -> list[PlotSpec] | list[ExportSpec]:
    root = Path(spec.experiment.root or ".").resolve()
    raw_steps: list[dict[str, Any]] = []

    for preset in presets or []:
        expanded = resolve_preset(preset)
        if not isinstance(expanded, list):
            raise ConfigError(f"{section}.presets '{preset}' did not resolve to a list of steps")
        for entry in expanded:
            if not isinstance(entry, dict):
                raise ConfigError(f"{section}.presets '{preset}' contains a non-mapping entry")
            enriched = dict(entry)
            enriched["preset_meta"] = {"preset": preset}
            raw_steps.append(enriched)

    for entry in specs or []:
        if isinstance(entry, StepSpec):
            raw_steps.append(entry.model_dump(by_alias=True))
        elif isinstance(entry, dict):
            raw_steps.append(entry)
        else:
            raise ConfigError(f"{section}.specs entry must be a mapping (got {type(entry).__name__})")

    if not raw_steps:
        return []

    default_reads = defaults.get("reads") or {}
    default_with = defaults.get("with") or {}
    if not isinstance(default_reads, dict):
        raise ConfigError(f"{section}.defaults.reads must be a mapping")
    if not isinstance(default_with, dict):
        raise ConfigError(f"{section}.defaults.with must be a mapping")

    finalized: list[dict[str, Any]] = []
    for step in raw_steps:
        step_id = step.get("id")
        if not step_id or not isinstance(step_id, str):
            raise ConfigError(f"Every {section} spec must include an id.")
        uses = step.get("uses")
        if not uses or not isinstance(uses, str):
            raise ConfigError(f"{section} {step_id}: uses must be a non-empty string")
        if "/" not in uses:
            raise ConfigError(f"{section} {step_id}: uses must be 'category/key'")
        category = uses.split("/", 1)[0]
        if section == "plots" and category != "plot":
            raise ConfigError(f"{section} {step_id}: uses must be plot/*")
        if section == "exports" and category != "export":
            raise ConfigError(f"{section} {step_id}: uses must be export/*")

        reads = step.get("reads") or {}
        if not isinstance(reads, dict):
            raise ConfigError(f"{section} {step_id}: reads must be a mapping")
        with_block = step.get("with") or {}
        if not isinstance(with_block, dict):
            raise ConfigError(f"{section} {step_id}: with must be a mapping")
        writes = step.get("writes") or {}
        if not isinstance(writes, dict):
            raise ConfigError(f"{section} {step_id}: writes must be a mapping")

        merged_reads = {**default_reads, **reads}
        merged_with = {**default_with, **with_block}

        preset_meta = step.get("preset_meta")
        finalized.append(
            {
                "id": step_id,
                "uses": uses,
                "reads": merged_reads,
                "with": merged_with,
                "writes": writes,
                "preset_meta": preset_meta if isinstance(preset_meta, dict) else None,
            }
        )

    if overrides:
        if not isinstance(overrides, dict):
            raise ConfigError(f"{section}.overrides must be a mapping of id -> overrides")
        ids = {s["id"] for s in finalized}
        unknown = sorted(set(overrides) - ids)
        if unknown:
            raise ConfigError(
                f"{section}.overrides reference unknown id(s): {unknown}. "
                "Check preset-expanded ids or remove stale overrides."
            )
        for i, step in enumerate(finalized):
            step_id = step["id"]
            if step_id not in overrides:
                continue
            merged = _deep_merge(step, overrides[step_id])
            if merged.get("id") != step_id:
                raise ConfigError(f"{section}.overrides for '{step_id}' cannot change the id.")
            finalized[i] = merged

    seen: set[str] = set()
    for step in finalized:
        step_id = step["id"]
        if step_id in seen:
            raise ConfigError(f"{section} contains duplicate spec id(s): {step_id}")
        seen.add(step_id)

    result: list[PlotSpec] | list[ExportSpec] = []
    for step in finalized:
        reads = step.get("reads") or {}
        if not isinstance(reads, dict):
            raise ConfigError(f"{section} {step['id']}: reads must be a mapping")
        writes = step.get("writes") or {}
        if not isinstance(writes, dict):
            raise ConfigError(f"{section} {step['id']}: writes must be a mapping")
        with_block = step.get("with") or {}
        if not isinstance(with_block, dict):
            raise ConfigError(f"{section} {step['id']}: with must be a mapping")
        normalized_reads: dict[str, str] = {}
        for key, value in reads.items():
            if isinstance(value, str) and value.startswith("file:"):
                raw = value.split("file:", 1)[1].strip()
                if not raw:
                    raise ConfigError(f"{section} {step['id']}: reads '{key}' uses an empty file: path.")
                path = Path(raw).expanduser()
                path = (root / path).resolve() if not path.is_absolute() else path.resolve()
                normalized_reads[key] = f"file:{path}"
            else:
                normalized_reads[key] = value
        result.append(
            spec_cls(
                id=step["id"],
                uses=step["uses"],
                reads=normalized_reads,
                with_=with_block,
                writes=writes,
                preset_meta=step.get("preset_meta"),
            )
        )
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
