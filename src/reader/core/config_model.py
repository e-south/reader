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
from pydantic import BaseModel, Field, field_validator

from reader.core.errors import ConfigError
from reader.core.presets import resolve_preset


class StepSpec(BaseModel):
    id: str
    uses: str
    reads: dict[str, str] = Field(default_factory=dict)  # input label -> artifact label OR "file:<path>"
    writes: dict[str, str] = Field(default_factory=dict)  # output label -> artifact label base
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class ReaderSpec(BaseModel):
    experiment: dict[str, Any]
    io: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    contracts: list[dict[str, Any]] = Field(default_factory=list)
    collections: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepSpec]
    reports: list[StepSpec] = Field(default_factory=list)

    @field_validator("experiment", mode="after")
    @classmethod
    def _validate_exp(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "outputs" not in v:
            raise ConfigError("experiment.outputs must be provided")
        return v

    @classmethod
    def load(cls, path: Path) -> ReaderSpec:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
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

        report_presets = data.get("report_presets", []) or []
        if not isinstance(report_presets, list):
            raise ConfigError("report_presets must be a list")
        report_overrides = data.get("report_overrides", {}) or {}
        if not isinstance(report_overrides, dict):
            raise ConfigError("report_overrides must be a mapping of id -> overrides")
        raw_reports = [{"preset": name} for name in report_presets]
        raw_reports.extend(data.get("reports", []) or [])
        data["reports"] = _expand_steps(raw_reports, section="reports")

        def _finalize_steps(steps: list[dict[str, Any]], overrides_map: dict[str, Any], *, section: str) -> None:
            for s in steps:
                step_id = s.get("id")
                if not step_id:
                    raise ConfigError(f"Every {section} step must include an id (after preset expansion).")
                if step_id in overrides_map:
                    s.update(_deep_merge(s, overrides_map[step_id]))
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

        _finalize_steps(data.get("steps", []), overrides, section="pipeline")
        _finalize_steps(data.get("reports", []), report_overrides, section="report")
        return cls.model_validate(data)
