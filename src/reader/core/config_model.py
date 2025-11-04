"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/config_model.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field, field_validator

from reader.core.errors import ConfigError


class StepSpec(BaseModel):
    id: str
    uses: str
    reads: Dict[str, str] = Field(default_factory=dict)    # input label -> artifact label OR "file:<path>"
    writes: Dict[str, str] = Field(default_factory=dict)   # output label -> artifact label base
    with_: Dict[str, Any] = Field(default_factory=dict, alias="with")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class ReaderSpec(BaseModel):
    experiment: Dict[str, Any]
    io: Dict[str, Any] = Field(default_factory=dict)
    runtime: Dict[str, Any] = Field(default_factory=dict)
    contracts: List[Dict[str, Any]] = Field(default_factory=list)
    collections: Dict[str, Any] = Field(default_factory=dict)
    steps: List[StepSpec]

    @field_validator("experiment", mode="after")
    @classmethod
    def _validate_exp(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if "outputs" not in v:
            raise ConfigError("experiment.outputs must be provided")
        return v

    @classmethod
    def load(cls, path: Path) -> "ReaderSpec":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Normalize relative paths to job file directory
        root = path.parent

        def _norm_io(d: Dict[str, Any]) -> Dict[str, Any]:
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
        for s in data.get("steps", []):
            # normalize file: pseudo-reads
            r = {}
            for k, v in s.get("reads", {}).items():
                if isinstance(v, str) and v.startswith("file:"):
                    p = v.split("file:", 1)[1]
                    r[k] = f"file:{(root / p).resolve()}"
                else:
                    r[k] = v
            s["reads"] = r
        return cls.model_validate(data)
