"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/cli/test_ux.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from reader.core import cli
from reader.core.config import ReaderSpec
from reader.core.engine import build_next_steps
from reader.core.registry import Plugin, PluginConfig, Registry
from reader.core.workbench import PluginSemantics
from reader.tests.support import base_reader_config, write_config


def _base_config() -> dict:
    return base_reader_config(
        experiment_id="exp",
        plot_specs=[{"id": "plot_a", "uses": "plot/time_series", "reads": {"df": "ingest/df"}}],
        export_specs=[
            {"id": "export_a", "uses": "export/csv", "reads": {"df": "ingest/df"}, "with": {"path": "a.csv"}}
        ],
    )


def test_ls_compact_name_column(monkeypatch, tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    long_name = "exp_" + ("x" * 80)
    exp_dir = exp_root / long_name
    exp_dir.mkdir(parents=True)
    write_config(exp_dir / "config.yaml", _base_config())

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
    assert any(cmd.startswith("reader records 1") for cmd in commands)
    assert any(cmd.startswith("reader plot 1") for cmd in commands)
    assert any("reader export 1" in cmd for cmd in commands)
    assert any(cmd.startswith("reader notebook 1") for cmd in commands)
    assert not any("--mode" in cmd for cmd in commands)
    assert not any("--edit" in cmd for cmd in commands)


def test_next_steps_prefers_config_notebook_preset() -> None:
    cfg = _base_config()
    cfg["notebooks"] = {"specs": [{"id": "basic", "uses": "notebook/basic"}]}
    spec = ReaderSpec.model_validate(cfg)
    steps = build_next_steps(spec, job_label="1")
    notes = [desc for _, desc in steps]
    assert any("template notebook/basic" in desc for desc in notes)


class _PluginCfg(PluginConfig):
    pass


class _PluginDummy(Plugin):
    key = "dummy"
    category = "plot"
    semantics = PluginSemantics(
        category="plot",
        domain="plate_reader",
        family="test_plot",
        summary="Synthetic plot plugin for CLI tests.",
    )
    ConfigModel = _PluginCfg

    @classmethod
    def input_contracts(cls):
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls):
        return {"files": "none"}

    def run(self, ctx, inputs, cfg):
        raise AssertionError("not used")


def test_plugins_command_shows_workbench_semantics(monkeypatch) -> None:
    reg = Registry()
    reg.register("plot", "dummy", _PluginDummy)
    test_console = Console(width=100, record=True, theme=cli.THEME, force_terminal=True)
    monkeypatch.setattr(cli, "console", test_console)
    monkeypatch.setattr(cli, "load_entry_points", lambda: reg)

    cli.plugins(category=None, domain=None, family=None)

    output = test_console.export_text()
    assert "plate_reader" in output
    assert "test_plot" in output
    assert "Synthetic plot plugin" in output
    assert "for CLI tests." in output
