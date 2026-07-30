from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

from reader.protocols.builtins import builtin_protocol_catalog
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl, build_workbench_decl


def write_config(target: Path, payload: dict | str) -> Path:
    path = target if target.suffix else target / "config.yaml"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(deepcopy(payload), sort_keys=False), encoding="utf-8")
    return path


def load_models(path: Path) -> tuple[ReaderSpec, WorkbenchDecl]:
    spec = ReaderSpec.load(path)
    return spec, build_workbench_decl(spec, source_path=path, protocols=builtin_protocol_catalog())


def build_decl(spec: ReaderSpec, *, source_path: Path | None = None) -> WorkbenchDecl:
    path = source_path or Path("/tmp/reader-test-config.yaml")
    return build_workbench_decl(spec, source_path=path, protocols=builtin_protocol_catalog())


def load_decl(path: Path) -> WorkbenchDecl:
    _, decl = load_models(path)
    return decl


def default_notebook_name() -> str:
    return f"EDA_{datetime.now().strftime('%Y%m%d')}.py"


def cytometry_test_gating_policy() -> dict:
    """Return one explicit, synthetic gating policy for protocol tests."""

    return {
        "cells_enabled": True,
        "cells_x_channel": "FSC-A",
        "cells_y_channel": "SSC-A",
        "cells_x_range": [100.0, 100_000.0],
        "cells_y_range": [100.0, 100_000.0],
        "singlets_enabled": True,
        "singlet_x_channel": "FSC-H",
        "singlet_y_channel": "FSC-A",
        "singlet_ratio_range": [0.8, 1.2],
        "fluorescence_channel": "GFP-A",
        "threshold_mode": "manual",
        "threshold_value": 1_000.0,
        "threshold_group_column": None,
        "threshold_control_value": None,
        "threshold_quantile": None,
        "group_column": "treatment",
        "minimum_final_events": 100,
        "minimum_final_percent": 1.0,
        "maximum_nonpositive_percent": 5.0,
        "nonpositive_scope": "all_events",
    }


def base_reader_config(
    *,
    experiment_id: str = "exp_001",
    title: str | None = None,
    lifecycle: str | None = None,
    protocol_id: str = "workbench/generic",
    protocol_inputs: dict | None = None,
    protocol_analysis: dict | None = None,
    protocol_outputs: dict | None = None,
    outputs: str = "./outputs",
    plotting: dict | None = None,
    annotations: dict | None = None,
    resources: dict | None = None,
) -> dict:
    payload = {
        "schema": "reader/v8",
        "experiment": {"id": experiment_id},
        "protocol": {
            "id": protocol_id,
            "inputs": deepcopy(protocol_inputs or {}),
            "analysis": deepcopy(protocol_analysis or {}),
            "outputs": deepcopy(protocol_outputs or {}),
        },
        "paths": {"outputs": outputs, "plots": "plots", "exports": "exports", "notebooks": "notebooks"},
    }
    if title is not None:
        payload["experiment"]["title"] = title
    if lifecycle is not None:
        payload["experiment"]["lifecycle"] = lifecycle
    if plotting is not None:
        payload["plotting"] = deepcopy(plotting)
    if annotations is not None:
        payload["annotations"] = deepcopy(annotations)
    if resources is not None:
        payload["resources"] = deepcopy(resources)
    return payload
