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
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator

from reader.core.errors import ConfigError
from reader.core.presets import expand_steps


class StepSpec(BaseModel):
    id: str
    uses: str
    reads: dict[str, str] = Field(default_factory=dict)  # input label -> artifact label OR "file:<path>"
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ExperimentSpec(BaseModel):
    id: str | None = None
    name: str | None = None
    outputs: str
    palette: str | None = "colorblind"
    plots_dir: str | None = "plots"
    root: str | None = None

    model_config = {"extra": "forbid"}

    @field_validator("outputs")
    @classmethod
    def _require_outputs(cls, v: str) -> str:
        if not str(v or "").strip():
            raise ValueError("experiment.outputs must be provided")
        return v


class CollectionsSpec(RootModel[dict[str, dict[str, list[dict[str, list[str]]]]]]):
    @field_validator("root")
    @classmethod
    def _validate_root(cls, v: dict[str, dict[str, list[dict[str, list[str]]]]]):
        for col, sets in v.items():
            if not isinstance(sets, dict):
                raise ValueError(f"collections.{col} must be a mapping of set_name -> list[dict]")
            for set_name, groups in sets.items():
                if not isinstance(groups, list):
                    raise ValueError(f"collections.{col}.{set_name} must be a list of single-key dicts")
                for item in groups:
                    if not isinstance(item, dict) or len(item) != 1:
                        raise ValueError(
                            f"collections.{col}.{set_name} entries must be single-key dicts like "
                            "{'Group': ['A','B']}"
                        )
                    _, members = next(iter(item.items()))
                    if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
                        raise ValueError(
                            f"collections.{col}.{set_name} entries must map to list[str] values"
                        )
        return v


class ReaderSpec(BaseModel):
    experiment: ExperimentSpec
    collections: CollectionsSpec | None = None
    presets: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepSpec]
    report_presets: list[str] = Field(default_factory=list)
    report_overrides: dict[str, Any] = Field(default_factory=dict)
    reports: list[StepSpec] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @classmethod
    def load(cls, path: Path) -> ReaderSpec:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            raise ConfigError(f"config.yaml is empty: {path}")
        # Normalize relative paths to job file directory
        root = path.parent

        data.setdefault("experiment", {})
        data["experiment"]["root"] = str(root.resolve())
        data["experiment"].setdefault("name", root.name)
        # Normalize experiment.outputs relative to config directory
        if "outputs" in data["experiment"]:
            outp = Path(str(data["experiment"]["outputs"]))
            if not outp.is_absolute():
                data["experiment"]["outputs"] = str((root / outp).resolve())
        else:
            raise ConfigError("experiment.outputs must be provided")

        data.setdefault("collections", {})

        # Expand presets (top-level and inline) and apply overrides.
        presets = data.pop("presets", []) or []
        overrides = data.pop("overrides", {}) or {}
        raw_steps = data.get("steps", []) or []
        data["steps"] = expand_steps(raw_steps=raw_steps, presets=presets, overrides=overrides)

        # Reports: optional post-pipeline steps (plots/exports).
        report_presets = data.pop("report_presets", []) or []
        report_overrides = data.pop("report_overrides", {}) or {}
        raw_reports = data.get("reports", []) or []
        data["reports"] = (
            expand_steps(raw_steps=raw_reports, presets=report_presets, overrides=report_overrides)
            if (raw_reports or report_presets)
            else []
        )

        if not data["steps"]:
            raise ConfigError("config.yaml defines no steps (use steps: or preset entries).")

        for bucket in ("steps", "reports"):
            for s in data.get(bucket, []):
                # normalize file: pseudo-reads
                r = {}
                for k, v in s.get("reads", {}).items():
                    if isinstance(v, str) and v.startswith("file:"):
                        p = v.split("file:", 1)[1]
                        r[k] = f"file:{(root / p).resolve()}"
                    else:
                        r[k] = v
                s["reads"] = r
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            parts = []
            for err in e.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", "invalid value")
                parts.append(f"{loc}: {msg}" if loc else msg)
            raise ConfigError("; ".join(parts)) from e
