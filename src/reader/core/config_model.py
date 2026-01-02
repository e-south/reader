"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/config_model.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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


class ReaderSpec(BaseModel):
    experiment: dict[str, Any]
    io: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    contracts: list[dict[str, Any]] = Field(default_factory=list)
    collections: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepSpec]
    deliverables: list[StepSpec] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @field_validator("experiment", mode="after")
    @classmethod
    def _validate_exp(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "outputs" not in v:
            raise ConfigError("experiment.outputs must be provided")
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
        legacy_keys = [k for k in ("reports", "report_presets", "report_overrides") if k in (data or {})]
        if legacy_keys:
            raise ConfigError(
                "Config uses legacy report keys. Rename to: "
                "deliverables, deliverable_presets, deliverable_overrides."
            )
        # Normalize relative paths to job file directory
        root = path.parent

        def _norm_io(d: dict[str, Any]) -> dict[str, Any]:
            def _fix(p):
                if isinstance(p, str) and (p.startswith("./") or p.startswith("../") or not Path(p).is_absolute()):
                    return str((root / p).resolve())
                return p

            return {k: _fix(v) for k, v in d.items()}

        data.setdefault("experiment", {})
        data["experiment"]["root"] = str(root.resolve())
        data["experiment"].setdefault("name", root.name)
        # Normalize experiment.outputs relative to config directory
        if "outputs" in data["experiment"]:
            outp = Path(str(data["experiment"]["outputs"]))
            if not outp.is_absolute():
                data["experiment"]["outputs"] = str((root / outp).resolve())
        if "io" in data:
            data["io"] = _norm_io(data["io"])

        overrides = data.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ConfigError("overrides must be a mapping of id -> overrides")

        def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
            out = dict(base)
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out

        def _expand_steps(raw_steps: list[dict[str, Any]], *, section: str) -> list[dict[str, Any]]:
            expanded: list[dict[str, Any]] = []
            for entry in raw_steps:
                if "preset" in entry:
                    preset_name = entry["preset"]
                    steps = resolve_preset(preset_name)
                    extra = {k: v for k, v in entry.items() if k != "preset"}
                    if extra:
                        if len(steps) != 1:
                            raise ConfigError(
                                f"Preset {preset_name!r} expands to {len(steps)} steps; "
                                f"inline overrides are only supported for single-step presets ({section})."
                            )
                        steps = [_deep_merge(steps[0], extra)]
                    expanded.extend(steps)
                else:
                    expanded.append(entry)
            return expanded

        raw_steps = data.get("steps", []) or []
        data["steps"] = _expand_steps(raw_steps, section="steps")

        deliverable_presets = data.get("deliverable_presets", []) or []
        if not isinstance(deliverable_presets, list):
            raise ConfigError("deliverable_presets must be a list")
        deliverable_overrides = data.get("deliverable_overrides", {}) or {}
        if not isinstance(deliverable_overrides, dict):
            raise ConfigError("deliverable_overrides must be a mapping of id -> overrides")
        raw_deliverables = [{"preset": name} for name in deliverable_presets]
        raw_deliverables.extend(data.get("deliverables", []) or [])
        data["deliverables"] = _expand_steps(raw_deliverables, section="deliverables")

        def _validate_override_keys(steps: list[dict[str, Any]], overrides_map: dict[str, Any], *, section: str) -> None:
            step_ids = {s.get("id") for s in steps if isinstance(s, dict)}
            unknown = sorted(set(overrides_map) - step_ids)
            if unknown:
                raise ConfigError(
                    f"{section} overrides reference unknown step id(s): {unknown}. "
                    "Check preset-expanded step ids or remove stale overrides."
                )

        def _finalize_steps(steps: list[dict[str, Any]], overrides_map: dict[str, Any], *, section: str) -> None:
            for s in steps:
                step_id = s.get("id")
                if not step_id:
                    raise ConfigError(f"Every {section} step must include an id (after preset expansion).")
                if step_id in overrides_map:
                    s.update(_deep_merge(s, overrides_map[step_id]))
                    if s.get("id") != step_id:
                        raise ConfigError(
                            f"{section} overrides for '{step_id}' cannot change the step id "
                            f"(got {s.get('id')!r})."
                        )
                # normalize file: pseudo-reads
                r = {}
                for k, v in s.get("reads", {}).items():
                    if isinstance(v, str) and v.startswith("file:"):
                        p = v.split("file:", 1)[1]
                        r[k] = f"file:{(root / p).resolve()}"
                    else:
                        r[k] = v
                s["reads"] = r
                s.setdefault("with", {})
                s.setdefault("writes", {})

        _validate_override_keys(data.get("steps", []), overrides, section="pipeline")
        _validate_override_keys(data.get("deliverables", []), deliverable_overrides, section="deliverables")
        _finalize_steps(data.get("steps", []), overrides, section="pipeline")
        _finalize_steps(data.get("deliverables", []), deliverable_overrides, section="deliverables")
        data.pop("overrides", None)
        data.pop("deliverable_overrides", None)
        data.pop("deliverable_presets", None)
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            details = "; ".join(
                f"{'.'.join(map(str, err.get('loc', [])))}: {err.get('msg')}" for err in e.errors()
            )
            raise ConfigError(f"Invalid config in {path}: {details}") from e
