from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from reader.tests.support import base_reader_config, load_decl, write_config
from reader.workbench.cli import app
from reader.workbench.cli.helpers import resolve_pipeline_step_id


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


def test_run_dry_run_allows_non_active_lifecycle(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, {**_run_config(), "experiment": {"id": "exp_run", "lifecycle": "draft"}})
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg), "--dry-run"])
    assert result.exit_code == 0
    assert "DRY RUN" in _plain(result.output)


def test_resolve_pipeline_step_id_hint_includes_target_config(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    decl = load_decl(cfg)

    with pytest.raises(typer.BadParameter, match="uv run reader steps") as exc_info:
        resolve_pipeline_step_id(decl, "missing_step", job_path=cfg)

    assert str(cfg) in str(exc_info.value)


def test_read_only_commands_do_not_create_journal(tmp_path: Path) -> None:
    cfg_payload = base_reader_config(
        experiment_id="exp_read_only",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}},
        protocol_analysis={"crosstalk_pairs": {"enabled": True, "export": True}},
        protocol_outputs={
            "plots": {"profile": "none", "include": ["raw_kinetics"]},
            "exports": {"include": ["crosstalk_pairs_table"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    cfg = write_config(tmp_path, cfg_payload)
    runner = CliRunner()

    commands = [
        ["explain", str(cfg)],
        ["validate", str(cfg), "--no-files"],
        ["run", str(cfg), "--dry-run"],
        ["plot", str(cfg), "--list"],
        ["plot", str(cfg), "--dry-run"],
        ["export", str(cfg), "--list"],
        ["export", str(cfg), "--dry-run"],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, _plain(result.output)

    assert not (tmp_path / "JOURNAL.md").exists()
    assert not (tmp_path / "journal.md").exists()
