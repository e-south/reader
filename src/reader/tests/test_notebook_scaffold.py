from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from reader.core.cli import app


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_plot_notebook_scaffold_uses_specs(tmp_path: Path) -> None:
    cfg = {
        "schema": "reader/v2",
        "experiment": {"id": "exp_nb"},
        "paths": {"outputs": "./outputs", "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {
            "specs": [
                {
                    "id": "plot_a",
                    "uses": "plot/time_series",
                    "reads": {"df": "ingest/df"},
                    "with": {"y": ["OD600"]},
                }
            ]
        },
        "exports": {"specs": []},
    }
    cfg_path = _write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg_path), "--mode", "notebook"])
    assert result.exit_code == 0
    nb_path = tmp_path / "notebooks" / "plots.py"
    assert nb_path.exists()
    content = nb_path.read_text(encoding="utf-8")
    assert "resolve_plot_specs" in content
    assert "plots_manifest.json" not in content
    assert "exports_manifest.json" not in content
