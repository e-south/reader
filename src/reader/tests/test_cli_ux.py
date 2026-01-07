"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_cli_ux.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console

from reader.core import cli
from reader.core.config_model import ReaderSpec
from reader.core.engine import build_next_steps


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_config() -> dict:
    return {
        "schema": "reader/v2",
        "experiment": {"id": "exp"},
        "paths": {"outputs": "./outputs", "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {"specs": [{"id": "plot_a", "uses": "plot/time_series", "reads": {"df": "ingest/df"}}]},
        "exports": {
            "specs": [{"id": "export_a", "uses": "export/csv", "reads": {"df": "ingest/df"}, "with": {"path": "a.csv"}}]
        },
    }


def test_ls_compact_name_column(monkeypatch, tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    long_name = "exp_" + ("x" * 80)
    exp_dir = exp_root / long_name
    exp_dir.mkdir(parents=True)
    _write_config(exp_dir / "config.yaml", _base_config())

    test_console = Console(width=60, record=True, theme=cli.THEME, force_terminal=True)
    monkeypatch.setattr(cli, "console", test_console)
    cli.ls(root=str(exp_root))

    output = test_console.export_text()
    max_line = max(len(line) for line in output.splitlines()) if output else 0
    assert max_line <= 80


def test_next_steps_commands_are_clean() -> None:
    spec = ReaderSpec.model_validate(_base_config())
    steps = build_next_steps(spec, job_label="1")
    commands = [cmd for cmd, _ in steps]
    assert any(cmd.startswith("reader artifacts 1") for cmd in commands)
    assert any(cmd.startswith("reader plot 1") for cmd in commands)
    assert any("reader export 1" in cmd for cmd in commands)
    assert any(cmd.startswith("reader notebook 1") for cmd in commands)
    assert not any("--mode" in cmd for cmd in commands)
    assert not any("--edit" in cmd for cmd in commands)


def test_next_steps_prefers_config_notebook_preset() -> None:
    cfg = _base_config()
    cfg["notebook"] = {"preset": "notebook/basic"}
    spec = ReaderSpec.model_validate(cfg)
    steps = build_next_steps(spec, job_label="1")
    notes = [desc for _, desc in steps]
    assert any("preset notebook/basic" in desc for desc in notes)
