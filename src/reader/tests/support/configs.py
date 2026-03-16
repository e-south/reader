from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl, build_workbench_decl


def write_config(target: Path, payload: dict | str) -> Path:
    path = target if target.suffix else target / "config.yaml"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(_normalize_test_config(payload), sort_keys=False), encoding="utf-8")
    return path


def load_models(path: Path) -> tuple[ReaderSpec, WorkbenchDecl]:
    spec = ReaderSpec.load(path)
    return spec, build_workbench_decl(spec, source_path=path)


def build_decl(spec: ReaderSpec, *, source_path: Path | None = None) -> WorkbenchDecl:
    path = source_path or Path("/tmp/reader-test-config.yaml")
    return build_workbench_decl(spec, source_path=path)


def load_decl(path: Path) -> WorkbenchDecl:
    _, decl = load_models(path)
    return decl


def default_notebook_name() -> str:
    return f"EDA_{datetime.now().strftime('%Y%m%d')}.py"


def base_reader_config(
    *,
    experiment_id: str = "exp_001",
    title: str | None = None,
    outputs: str = "./outputs",
    pipeline_steps: list[dict] | None = None,
    plot_specs: list[dict] | None = None,
    export_specs: list[dict] | None = None,
    notebook_specs: list[dict] | None = None,
    plotting: dict | None = None,
    assay: dict | None = None,
) -> dict:
    payload = {
        "schema": "reader/v4",
        "experiment": {"id": experiment_id},
        "paths": {"outputs": outputs, "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": pipeline_steps or [{"id": "ingest", "plugin": "ingest/synergy_h1"}]},
        "plots": {"specs": plot_specs or []},
        "exports": {"specs": export_specs or []},
    }
    if title is not None:
        payload["experiment"]["title"] = title
    if notebook_specs is not None:
        payload["notebooks"] = {"specs": notebook_specs}
    if plotting is not None:
        payload["plotting"] = deepcopy(plotting)
    if assay is not None:
        payload["assay"] = deepcopy(assay)
    return payload


def _normalize_test_config(payload: dict) -> dict:
    data = deepcopy(payload)
    for section, label in (("pipeline", "steps"), ("plots", "specs"), ("exports", "specs")):
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        defaults = block.get("defaults")
        if isinstance(defaults, dict) and isinstance(defaults.get("reads"), dict):
            defaults["reads"] = {key: _normalize_input_binding(value) for key, value in defaults["reads"].items()}
        entries = block.get(label)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("reads"), dict):
                entry["reads"] = {key: _normalize_input_binding(value) for key, value in entry["reads"].items()}
            if isinstance(entry.get("writes"), dict):
                entry["writes"] = {key: _normalize_output_binding(value) for key, value in entry["writes"].items()}
    return data


def _normalize_input_binding(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    return {"record": value}


def _normalize_output_binding(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    return {"record": value}
