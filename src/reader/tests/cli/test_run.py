from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from reader.tests.support import base_reader_config, write_config
from reader.workbench.cli import app


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
