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
from types import SimpleNamespace

from typer.testing import CliRunner

import reader.workbench.cli as cli
from reader.tests.support import base_reader_config, default_notebook_name, write_config
from reader.workbench.cli import app


def _base_config() -> dict:
    return base_reader_config(experiment_id="exp_nb")


def test_notebook_defaults_to_edit_mode(monkeypatch, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    called: dict[str, object] = {}

    def _fake_launch(
        mode: str, target: Path, *, has_fcs: bool, headless: bool = False, port: int | None = None
    ) -> None:
        called["mode"] = mode
        called["target"] = target

    monkeypatch.setattr(cli.notebook_commands, "_launch_marimo", _fake_launch)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path)])
    assert result.exit_code == 0
    assert called.get("mode") == "edit"


def test_notebook_mode_none_skips_launch(monkeypatch, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())

    def _fail_launch(*args, **kwargs) -> None:
        raise AssertionError("launch should not be called")

    monkeypatch.setattr(cli.notebook_commands, "_launch_marimo", _fail_launch)
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

    monkeypatch.setattr(cli.notebook_commands, "_launch_marimo", _fail_launch)
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

    monkeypatch.setattr(cli.notebook_commands, "_launch_marimo", _fail_launch)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    assert "SFXI" in nb_path.read_text(encoding="utf-8")


def test_notebook_launch_failure_prints_help(monkeypatch, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    original_load = cli.notebook_commands._load

    class _Proc:
        pid = 1234

        def wait(self):
            return 1

    launch_calls: dict[str, object] = {}

    def _fake_plan_marimo_launch(**kwargs):
        return SimpleNamespace(
            cmd=(sys.executable, "-m", "marimo", "edit", str(kwargs["target"])),
            env={},
            url="http://127.0.0.1:2718",
            port=2718,
            host="127.0.0.1",
            target=kwargs["target"],
            repo_root=tmp_path,
            runtime_paths=SimpleNamespace(
                root=tmp_path / ".cache" / "marimo", registry_path=tmp_path / "sessions.json"
            ),
            reused_session=None,
            terminated_sessions=(),
        )

    def _fake_load(name: str):
        if name == "reader.workbench.notebooks.launch":
            return SimpleNamespace(
                plan_marimo_launch=_fake_plan_marimo_launch,
                register_managed_session=lambda **kwargs: launch_calls.setdefault("registered", kwargs),
                unregister_managed_session=lambda **kwargs: launch_calls.setdefault("unregistered", kwargs),
                open_url=lambda url: None,
            )
        return original_load(name)

    def _fake_popen(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr(cli.notebook_commands, "_load", _fake_load)
    monkeypatch.setattr(cli.subprocess, "Popen", _fake_popen)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path)])
    assert result.exit_code == 1
    assert "Could not launch marimo automatically." in result.output
    assert "uv sync --locked --group notebooks" in result.output


def test_launch_marimo_uses_active_interpreter(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, object] = {}
    original_load = cli.notebook_commands._load

    class _Proc:
        pid = 4321

        def wait(self):
            return 0

    def _fake_plan_marimo_launch(**kwargs):
        return SimpleNamespace(
            cmd=(
                sys.executable,
                "-m",
                "marimo",
                "edit",
                "--host",
                "127.0.0.1",
                "--port",
                "2718",
                str(kwargs["target"]),
            ),
            env={
                "PYTHONPATH": str(Path(__file__).resolve().parents[4]),
                "XDG_CONFIG_HOME": str(tmp_path / ".cache" / "marimo" / "xdg-config"),
            },
            url="http://127.0.0.1:2718",
            port=2718,
            host="127.0.0.1",
            target=kwargs["target"],
            repo_root=tmp_path,
            runtime_paths=SimpleNamespace(
                root=tmp_path / ".cache" / "marimo", registry_path=tmp_path / "sessions.json"
            ),
            reused_session=None,
            terminated_sessions=(),
        )

    def _fake_load(name: str):
        if name == "reader.workbench.notebooks.launch":
            return SimpleNamespace(
                plan_marimo_launch=_fake_plan_marimo_launch,
                register_managed_session=lambda **kwargs: called.setdefault("registered", kwargs),
                unregister_managed_session=lambda **kwargs: called.setdefault("unregistered", kwargs),
                open_url=lambda url: called.setdefault("opened_url", url),
            )
        return original_load(name)

    def _fake_popen(cmd, env=None):
        called["cmd"] = cmd
        called["env"] = env
        return _Proc()

    monkeypatch.setattr(cli.notebook_commands, "_load", _fake_load)
    monkeypatch.setattr(cli.subprocess, "Popen", _fake_popen)
    cli._launch_marimo("edit", tmp_path / "notebook.py", has_fcs=False)
    cmd = called.get("cmd")
    assert isinstance(cmd, tuple)
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ("-m", "marimo")
    env = called.get("env")
    assert isinstance(env, dict)
    assert "READER_MARIMO_RUNTIME_PATCH" not in env
    assert "XDG_CONFIG_HOME" in env
    assert "registered" in called
    assert "unregistered" in called
