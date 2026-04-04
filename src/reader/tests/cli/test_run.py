from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from reader.tests.support import base_reader_config, write_config
from reader.workbench.cli import app


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def _run_config() -> dict:
    return base_reader_config(
        experiment_id="exp_run",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "labels": {
                "design_id": {
                    "source": "design_id",
                    "output": "design_id_alias",
                    "values": {},
                }
            }
        },
    )


def test_run_rejects_reversed_range(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg), "--from", "labels", "--until", "ingest", "--dry-run"])
    assert result.exit_code == 1
    assert "--from 'labels' comes after --until 'ingest'" in result.output


def test_run_dry_run_json_surfaces_slice(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg), "--dry-run", "--from", "labels", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["slice"]["from"] == "labels"
    assert payload["implementation"]["plan"]["pipeline_flow"][0] == "labels"
    assert payload["implementation"]["compiled"]["plots"] == []
    assert payload["implementation"]["compiled"]["exports"] == []


def test_run_json_requires_dry_run(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg), "--format", "json"])
    assert result.exit_code != 0
    assert "only supported with --dry-run" in _plain(result.output)
