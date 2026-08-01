from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import reader_workbench.workbench.cli as cli
from reader_workbench.tests.support import (
    base_reader_config,
    cytometry_test_gating_policy,
    default_notebook_name,
    write_config,
)
from reader_workbench.workbench.cli import app


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


def test_notebook_overwrite_is_explicit_and_noninteractive(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    runner = CliRunner()
    create = runner.invoke(
        app,
        ["notebook", str(cfg_path), "--mode", "none", "--name", "review.py"],
    )
    assert create.exit_code == 0
    target = tmp_path / "outputs" / "notebooks" / "review.py"
    target.write_text("stale\n", encoding="utf-8")

    overwrite = runner.invoke(
        app,
        ["notebook", str(cfg_path), "--mode", "none", "--name", "review.py", "--overwrite"],
    )

    assert overwrite.exit_code == 0
    assert "Overwrite?" not in overwrite.output
    assert "Notebook overwritten" in overwrite.output
    assert target.read_text(encoding="utf-8") != "stale\n"


def test_notebook_rejects_symlinked_notebooks_root_before_write(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (outputs / "notebooks").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    result = CliRunner().invoke(app, ["notebook", str(cfg_path), "--mode", "none"])

    assert result.exit_code != 0
    assert "notebooks" in result.output
    assert list(outside.iterdir()) == []


def test_notebook_overwrite_rejects_symlink_target(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    notebooks = tmp_path / "outputs" / "notebooks"
    notebooks.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("original", encoding="utf-8")
    target = notebooks / "review.py"
    try:
        target.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    result = CliRunner().invoke(
        app,
        ["notebook", str(cfg_path), "--mode", "none", "--name", "review.py", "--overwrite"],
        input="y\n",
    )

    assert result.exit_code != 0
    assert "symlink" in result.output.lower()
    assert outside.read_text(encoding="utf-8") == "original"


def test_notebook_existing_symlink_target_is_rejected_before_launch(monkeypatch, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    notebooks = tmp_path / "outputs" / "notebooks"
    notebooks.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("original", encoding="utf-8")
    target = notebooks / "review.py"
    try:
        target.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    def _fail_launch(*args, **kwargs) -> None:
        raise AssertionError("launch should not be called")

    monkeypatch.setattr(cli.notebook_commands, "_launch_marimo", _fail_launch)
    result = CliRunner().invoke(app, ["notebook", str(cfg_path), "--name", "review.py"])

    assert result.exit_code != 0
    assert "symlink" in result.output.lower()
    assert outside.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("name", ["", ".", "..", "review.txt", "nested/review.py", "../../escaped.py"])
def test_notebook_name_must_be_one_nonempty_filename(name: str, tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())

    result = CliRunner().invoke(app, ["notebook", str(cfg_path), "--mode", "none", "--name", name])

    assert result.exit_code != 0
    assert "non-empty .py filename" in result.output
    assert not (tmp_path / "escaped.py").exists()


def test_notebook_name_rejects_absolute_path(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    outside = tmp_path / "outside.py"

    result = CliRunner().invoke(
        app,
        ["notebook", str(cfg_path), "--mode", "none", "--name", str(outside)],
    )

    assert result.exit_code != 0
    assert "non-empty .py filename" in result.output
    assert not outside.exists()


def test_notebook_auto_selects_canonical_eda_for_cytometry_protocol(monkeypatch, tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_nb",
        protocol_id="cytometry/flow_panel",
        protocol_inputs={"gating": cytometry_test_gating_policy()},
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
    content = nb_path.read_text(encoding="utf-8")
    assert "collect_notebook_deliverables" in content
    assert "notebook/cytometry" not in content


def test_notebook_auto_selects_canonical_eda_for_four_state_vector_protocol(monkeypatch, tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_nb",
        protocol_id="logic/four_state_vector_screen",
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
    notebook = nb_path.read_text(encoding="utf-8")
    assert "build_notebook_deliverable_selector" in notebook
    assert "four-state vector 8-vector review" not in notebook


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
        if name == "reader_workbench.workbench.notebooks.launch":
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
    assert "does not publish a notebook dependency extra" in result.output
    assert "separately managed" in result.output
    assert "audited environment" in result.output


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
        if name == "reader_workbench.workbench.notebooks.launch":
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
