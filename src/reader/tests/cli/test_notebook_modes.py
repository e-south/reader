"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/cli/test_notebook_modes.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

import reader.core.cli as cli
from reader.core.cli import app
from reader.tests.support import base_reader_config, default_notebook_name, write_config


def _base_config() -> dict:
    return base_reader_config(experiment_id="exp_nb")


def test_notebook_defaults_to_edit_mode(monkeypatch, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    called: dict[str, object] = {}

    def _fake_launch(mode: str, target: Path, *, has_fcs: bool) -> None:
        called["mode"] = mode
        called["target"] = target

    monkeypatch.setattr(cli, "_launch_marimo", _fake_launch)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path)])
    assert result.exit_code == 0
    assert called.get("mode") == "edit"


def test_notebook_mode_none_skips_launch(monkeypatch, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())

    def _fail_launch(*args, **kwargs) -> None:
        raise AssertionError("launch should not be called")

    monkeypatch.setattr(cli, "_launch_marimo", _fail_launch)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    assert default_notebook_name() in result.output


def test_notebook_launch_failure_prints_help(monkeypatch, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())

    class _Result:
        returncode = 1

    def _fake_run(*args, **kwargs):
        return _Result()

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path)])
    assert result.exit_code == 1
    assert "Could not launch marimo automatically." in result.output
    assert "uv sync --locked --group notebooks" in result.output


def test_launch_marimo_uses_active_interpreter(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, object] = {}

    class _Result:
        returncode = 0

    def _fake_run(cmd, check=False):
        called["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    cli._launch_marimo("edit", tmp_path / "notebook.py", has_fcs=False)
    cmd = called.get("cmd")
    assert isinstance(cmd, list)
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "marimo"]
