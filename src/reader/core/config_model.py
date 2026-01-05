"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/config_model.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from reader.core.errors import ConfigError
from reader.core.presets import resolve_preset


class StepSpec(BaseModel):
    id: str
    uses: str
    reads: dict[str, str] = Field(default_factory=dict)  # input label -> artifact label OR "file:<path>"
    writes: dict[str, str] = Field(default_factory=dict)  # output label -> artifact label
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ExperimentSpec(BaseModel):
    id: str
    title: str | None = None
    root: str | None = None  # internal: experiment directory

    model_config = {"extra": "forbid"}

    @field_validator("id", mode="after")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ConfigError("experiment.id must be a non-empty string")
        return v


class PathsSpec(BaseModel):
    outputs: str = "./outputs"
    plots: str = "plots"
    exports: str = "exports"
    notebooks: str = "notebooks"

    model_config = {"extra": "forbid"}


class PlottingSpec(BaseModel):
    palette: str | None = "colorblind"

    model_config = {"extra": "forbid"}


class DataSpec(BaseModel):
    groupings: dict[str, Any] = Field(default_factory=dict)
    aliases: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class PipelineSpec(BaseModel):
    presets: list[str] = Field(default_factory=list)
    runtime: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepSpec]

    model_config = {"extra": "forbid"}


class SpecDefaults(BaseModel):
    reads: dict[str, str] = Field(default_factory=dict)
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")

    model_config = {"extra": "forbid"}


class PlotSection(BaseModel):
    presets: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    defaults: SpecDefaults = Field(default_factory=SpecDefaults)
    specs: list[StepSpec] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ExportSection(BaseModel):
    presets: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    defaults: SpecDefaults = Field(default_factory=SpecDefaults)
    specs: list[StepSpec] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class NotebookSpec(BaseModel):
    preset: str | None = None

    model_config = {"extra": "forbid"}


class ReaderSpec(BaseModel):
    schema_: str = Field(alias="schema")
    experiment: ExperimentSpec
    paths: PathsSpec = Field(default_factory=PathsSpec)
    plotting: PlottingSpec = Field(default_factory=PlottingSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    pipeline: PipelineSpec
    plots: PlotSection = Field(default_factory=PlotSection)
    exports: ExportSection = Field(default_factory=ExportSection)
    notebook: NotebookSpec = Field(default_factory=NotebookSpec)

    model_config = {"extra": "forbid"}

    @field_validator("schema_", mode="after")
    @classmethod
    def _validate_schema(cls, v: str) -> str:
        if v != "reader/v2":
            raise ConfigError("Config schema must be 'reader/v2'. This repo only supports reader/v2.")
        return v

    @classmethod
    def load(cls, path: Path) -> ReaderSpec:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError(
                f"Config must be a mapping (YAML object) in {path}. "
                "Check for empty files or top-level lists."
            )
        schema = data.get("schema")
        if schema != "reader/v2":
            raise ConfigError(
                f"Config schema must be 'reader/v2'. This repo only supports reader/v2 (found {schema!r})."
            )

        legacy_keys = {
            "steps",
            "overrides",
            "collections",
            "deliverables",
            "deliverable_presets",
            "deliverable_overrides",
        }
        exp_legacy = {"name", "outputs", "plots_dir", "palette"}
        illegal = sorted(k for k in legacy_keys if k in data)
        if "experiment" in data and isinstance(data["experiment"], dict):
            illegal_exp = sorted(k for k in exp_legacy if k in data["experiment"])
        else:
            illegal_exp = []
        if illegal or illegal_exp:
            parts = []
            if illegal:
                parts.append(f"top-level keys: {illegal}")
            if illegal_exp:
                parts.append(f"experiment keys: {illegal_exp}")
            raise ConfigError(
                "Legacy v1 config keys are not supported in reader/v2. "
                "Remove/replace: " + "; ".join(parts)
            )

        if "experiment" not in data or not isinstance(data["experiment"], dict):
            raise ConfigError("experiment must be a mapping with required keys: id (and optional title)")
        if "pipeline" not in data or not isinstance(data["pipeline"], dict):
            raise ConfigError("pipeline must be a mapping and include steps")
        if "steps" not in data["pipeline"]:
            raise ConfigError("pipeline.steps is required (use an empty list if there are no pipeline steps).")

        root = path.parent.resolve()

        def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
            out = dict(base)
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out

        def _ensure_step_list(raw_steps: Any, *, section: str, label: str) -> list[dict[str, Any]]:
            if not isinstance(raw_steps, list):
                raise ConfigError(f"{section}.{label} must be a list")
            normalized: list[dict[str, Any]] = []
            for i, entry in enumerate(raw_steps, 1):
                if not isinstance(entry, dict):
                    raise ConfigError(f"{section}.{label} entry #{i} must be a mapping")
                if "preset" in entry:
                    raise ConfigError(
                        f"{section}.{label} does not support inline preset expansion; "
                        f"use {section}.presets instead."
                    )
                normalized.append(entry)
            return normalized

        def _expand_presets(preset_names: list[str], *, section: str) -> list[dict[str, Any]]:
            expanded: list[dict[str, Any]] = []
            for name in preset_names:
                steps = resolve_preset(name)
                expanded.extend(steps)
            return expanded

        def _validate_overrides(steps: list[dict[str, Any]], overrides_map: dict[str, Any], *, section: str) -> None:
            if not isinstance(overrides_map, dict):
                raise ConfigError(f"{section}.overrides must be a mapping of id -> overrides")
            step_ids = {s.get("id") for s in steps if isinstance(s, dict)}
            unknown = sorted(set(overrides_map) - step_ids)
            if unknown:
                raise ConfigError(
                    f"{section}.overrides reference unknown step id(s): {unknown}. "
                    "Check preset-expanded step ids or remove stale overrides."
                )

        def _validate_step_kind(step: dict[str, Any], *, section: str) -> None:
            uses = str(step.get("uses", "") or "")
            if "/" not in uses:
                raise ConfigError(f"{section} {step.get('id', '<missing>')}: uses must be 'category/key'")
            category = uses.split("/", 1)[0]
            if section == "pipeline" and category in {"plot", "export"}:
                raise ConfigError(f"pipeline {step.get('id')}: plot/export plugins are not allowed in pipeline.")
            if section == "plots" and category != "plot":
                raise ConfigError(f"plots {step.get('id')}: uses must be plot/*")
            if section == "exports" and category != "export":
                raise ConfigError(f"exports {step.get('id')}: uses must be export/*")

        def _finalize_steps(
            steps: list[dict[str, Any]], overrides_map: dict[str, Any], *, section: str
        ) -> list[dict[str, Any]]:
            finalized: list[dict[str, Any]] = []
            for s in steps:
                step_id = s.get("id")
                if not step_id or not isinstance(step_id, str):
                    raise ConfigError(f"Every {section} step must include an id.")
                if step_id in overrides_map:
                    merged = _deep_merge(s, overrides_map[step_id])
                    if merged.get("id") != step_id:
                        raise ConfigError(
                            f"{section}.overrides for '{step_id}' cannot change the step id (got {merged.get('id')!r})."
                        )
                    s = merged
                reads_raw = s.get("reads", {})
                if reads_raw is None:
                    reads_raw = {}
                if not isinstance(reads_raw, dict):
                    raise ConfigError(f"{section} {step_id}: reads must be a mapping")
                writes_raw = s.get("writes", {})
                if writes_raw is None:
                    writes_raw = {}
                if not isinstance(writes_raw, dict):
                    raise ConfigError(f"{section} {step_id}: writes must be a mapping")
                with_raw = s.get("with", {})
                if with_raw is None:
                    with_raw = {}
                if not isinstance(with_raw, dict):
                    raise ConfigError(f"{section} {step_id}: with must be a mapping")
                # normalize file: pseudo-reads
                normalized_reads: dict[str, str] = {}
                for k, v in reads_raw.items():
                    if isinstance(v, str) and v.startswith("file:"):
                        raw = v.split("file:", 1)[1].strip()
                        if not raw:
                            raise ConfigError(f"{section} {step_id}: reads '{k}' uses an empty file: path.")
                        p = Path(raw).expanduser()
                        p = (root / p).resolve() if not p.is_absolute() else p.resolve()
                        normalized_reads[k] = f"file:{p}"
                    else:
                        normalized_reads[k] = v
                s["reads"] = normalized_reads
                s.setdefault("with", {})
                s.setdefault("writes", {})
                _validate_step_kind(s, section=section)
                finalized.append(s)
            return finalized

        def _ensure_unique_ids(steps: list[dict[str, Any]], *, section: str) -> set[str]:
            ids = [s.get("id") for s in steps if isinstance(s, dict)]
            dupes = sorted(k for k, v in Counter(ids).items() if k and v > 1)
            if dupes:
                raise ConfigError(f"{section} contains duplicate step id(s): {dupes}")
            return {i for i in ids if i}

        # Normalize optional sections
        data.setdefault("paths", {})
        if not isinstance(data["paths"], dict):
            raise ConfigError("paths must be a mapping")
        data.setdefault("plotting", {})
        if not isinstance(data["plotting"], dict):
            raise ConfigError("plotting must be a mapping")
        data.setdefault("data", {})
        if not isinstance(data["data"], dict):
            raise ConfigError("data must be a mapping")
        data.setdefault("plots", {})
        if not isinstance(data["plots"], dict):
            raise ConfigError("plots must be a mapping")
        data.setdefault("exports", {})
        if not isinstance(data["exports"], dict):
            raise ConfigError("exports must be a mapping")
        if "steps" in data["plots"]:
            raise ConfigError("plots.steps is not supported in reader/v2. Use plots.specs.")
        if "steps" in data["exports"]:
            raise ConfigError("exports.steps is not supported in reader/v2. Use exports.specs.")
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

        # Normalize paths
        outputs_raw = data["paths"].get("outputs", "./outputs")
        if not isinstance(outputs_raw, str) or not outputs_raw.strip():
            raise ConfigError("paths.outputs must be a non-empty string path")
        outputs_path = Path(outputs_raw).expanduser()
        if not outputs_path.is_absolute():
            outputs_path = (root / outputs_path).resolve()
        data["paths"]["outputs"] = str(outputs_path)
        plots_subdir = data["paths"].get("plots", "plots")
        exports_subdir = data["paths"].get("exports", "exports")
        notebooks_subdir = data["paths"].get("notebooks", "notebooks")
        for key, val in (("plots", plots_subdir), ("exports", exports_subdir), ("notebooks", notebooks_subdir)):
            if val is None:
                raise ConfigError(f"paths.{key} must be a string subdirectory (use '.' to flatten).")
            if not isinstance(val, str):
                raise ConfigError(f"paths.{key} must be a string subdirectory")
            if Path(val).is_absolute():
                raise ConfigError(f"paths.{key} must be relative to paths.outputs, not absolute.")

        # plotting.palette validation
        palette_raw = data["plotting"].get("palette", None)
        if palette_raw is not None and (not isinstance(palette_raw, str) or not palette_raw.strip()):
            raise ConfigError("plotting.palette must be a non-empty string or null")

        # data.groupings / data.aliases validation
        groupings_raw = data["data"].get("groupings", {}) or {}
        if not isinstance(groupings_raw, dict):
            raise ConfigError("data.groupings must be a mapping")
        aliases_raw = data["data"].get("aliases", {}) or {}
        if not isinstance(aliases_raw, dict):
            raise ConfigError("data.aliases must be a mapping")
        data["data"]["groupings"] = groupings_raw
        data["data"]["aliases"] = aliases_raw

        # pipeline runtime validation
        runtime_raw = data["pipeline"].get("runtime", {}) or {}
        if not isinstance(runtime_raw, dict):
            raise ConfigError("pipeline.runtime must be a mapping")
        if "strict" in runtime_raw and not isinstance(runtime_raw["strict"], bool):
            raise ConfigError("pipeline.runtime.strict must be a boolean (true/false)")
        data["pipeline"]["runtime"] = runtime_raw

        # presets + specs validation
        pipeline_presets = data["pipeline"].get("presets", []) or []
        if not isinstance(pipeline_presets, list):
            raise ConfigError("pipeline.presets must be a list")
        plots_presets = data["plots"].get("presets", []) or []
        if not isinstance(plots_presets, list):
            raise ConfigError("plots.presets must be a list")
        exports_presets = data["exports"].get("presets", []) or []
        if not isinstance(exports_presets, list):
            raise ConfigError("exports.presets must be a list")

        pipeline_steps = _expand_presets(pipeline_presets, section="pipeline")
        pipeline_steps.extend(
            _ensure_step_list(data["pipeline"].get("steps", []) or [], section="pipeline", label="steps")
        )
        plots_specs = _ensure_step_list(data["plots"].get("specs", []) or [], section="plots", label="specs")
        exports_specs = _ensure_step_list(data["exports"].get("specs", []) or [], section="exports", label="specs")

        pipeline_overrides = data["pipeline"].get("overrides", {}) or {}
        _validate_overrides(pipeline_steps, pipeline_overrides, section="pipeline")

        pipeline_steps = _finalize_steps(pipeline_steps, pipeline_overrides, section="pipeline")
        _ensure_unique_ids(pipeline_steps, section="pipeline")

        data["pipeline"]["steps"] = pipeline_steps
        data["plots"]["specs"] = plots_specs
        data["exports"]["specs"] = exports_specs

        # set experiment.root for internal use
        data["experiment"]["root"] = str(root)

        try:
            return cls.model_validate(data)
        except ValidationError as e:
            details = "; ".join(
                f"{'.'.join(map(str, err.get('loc', [])))}: {err.get('msg')}" for err in e.errors()
            )
            raise ConfigError(f"Invalid config in {path}: {details}") from e
