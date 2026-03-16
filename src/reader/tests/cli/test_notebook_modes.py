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

import reader.workbench.cli as cli
from reader.tests.support import base_reader_config, default_notebook_name, write_config
from reader.workbench.cli import app


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


def test_notebook_auto_selects_cytometry_preset_from_protocol(monkeypatch, tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_nb",
        protocol_id="cytometry/flow_panel",
        resources={"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}},
    )
    cfg_path = write_config(tmp_path, cfg)

    def _fail_launch(*args, **kwargs) -> None:
        raise AssertionError("launch should not be called")

    monkeypatch.setattr(cli, "_launch_marimo", _fail_launch)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    assert "Cytometry" in nb_path.read_text(encoding="utf-8")


def test_notebook_auto_selects_sfxi_template_from_protocol(monkeypatch, tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_nb",
        protocol_id="logic/sfxi_screen",
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    cfg_path = write_config(tmp_path, cfg)

    def _fail_launch(*args, **kwargs) -> None:
        raise AssertionError("launch should not be called")

    monkeypatch.setattr(cli, "_launch_marimo", _fail_launch)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    assert "SFXI" in nb_path.read_text(encoding="utf-8")


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
