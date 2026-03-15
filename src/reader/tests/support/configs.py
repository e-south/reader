from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


def write_config(target: Path, payload: dict | str) -> Path:
    path = target if target.suffix else target / "config.yaml"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


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
    semantics: dict | None = None,
) -> dict:
    payload = {
        "schema": "reader/v3",
        "experiment": {"id": experiment_id},
        "paths": {"outputs": outputs, "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": pipeline_steps or [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {"specs": plot_specs or []},
        "exports": {"specs": export_specs or []},
    }
    if title is not None:
        payload["experiment"]["title"] = title
    if notebook_specs is not None:
        payload["notebooks"] = {"specs": notebook_specs}
    if plotting is not None:
        payload["plotting"] = deepcopy(plotting)
    if semantics is not None:
        payload["semantics"] = deepcopy(semantics)
    return payload
