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
from typer.testing import CliRunner

from reader.contracts import builtin_contract_catalog
from reader.runtime import ReaderRuntime
from reader.tests.support import base_reader_config, build_decl, write_config
from reader.workbench import PluginSemantics, cli
from reader.workbench.assets import AssetCatalog, build_plugin_asset
from reader.workbench.config import ReaderSpec
from reader.workbench.engine import build_next_steps
from reader.workbench.ports import dataframe_input, file_bundle_output
from reader.workbench.registry import Plugin, PluginConfig, Registry


def _base_config() -> dict:
    return base_reader_config(
        experiment_id="exp",
        plot_specs=[{"id": "plot_a", "plugin": "plot/time_series", "reads": {"df": {"record": "ingest/df"}}}],
        export_specs=[
            {
                "id": "export_a",
                "plugin": "export/csv",
                "reads": {"df": {"record": "ingest/df"}},
                "with": {"path": "a.csv"},
            }
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
    cli.ls(root=str(exp_root), include_scaffolds=False)

    output = test_console.export_text()
    max_line = max(len(line) for line in output.splitlines()) if output else 0
    assert max_line <= 80


def test_ls_excludes_template_dirs_by_default(monkeypatch, tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    year_dir = exp_root / "2025" / "real_exp"
    template_dir = exp_root / "template"
    year_dir.mkdir(parents=True)
    template_dir.mkdir(parents=True)
    write_config(year_dir / "config.yaml", _base_config())
    write_config(template_dir / "config.yaml", _base_config())

    test_console = Console(width=80, record=True, theme=cli.THEME, force_terminal=True)
    monkeypatch.setattr(cli, "console", test_console)
    cli.ls(root=str(exp_root), include_scaffolds=False)

    output = test_console.export_text()
    assert "real_exp" in output
    assert "template" not in output


def test_numeric_job_index_ignores_template_dirs(monkeypatch, tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    template_dir = exp_root / "template"
    year_dir = exp_root / "2025" / "real_exp"
    template_dir.mkdir(parents=True)
    year_dir.mkdir(parents=True)
    write_config(template_dir / "config.yaml", _base_config())
    write_config(year_dir / "config.yaml", _base_config())

    monkeypatch.chdir(tmp_path)
    assert cli._infer_job_path("1") == (year_dir / "config.yaml").resolve()


def test_ls_all_includes_template_dirs(monkeypatch, tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    year_dir = exp_root / "2025" / "real_exp"
    template_dir = exp_root / "template"
    year_dir.mkdir(parents=True)
    template_dir.mkdir(parents=True)
    write_config(year_dir / "config.yaml", _base_config())
    write_config(template_dir / "config.yaml", _base_config())

    test_console = Console(width=80, record=True, theme=cli.THEME, force_terminal=True)
    monkeypatch.setattr(cli, "console", test_console)
    cli.ls(root=str(exp_root), include_scaffolds=True)

    output = test_console.export_text()
    assert "real_exp" in output
    assert "template" in output


def test_next_steps_commands_are_clean() -> None:
    spec = ReaderSpec.model_validate(_base_config())
    steps = build_next_steps(build_decl(spec), job_label="1")
    commands = [cmd for cmd, _ in steps]
    assert any(cmd.startswith("reader records 1") for cmd in commands)
    assert any(cmd.startswith("reader plot 1") for cmd in commands)
    assert any("reader export 1" in cmd for cmd in commands)
    assert any(cmd.startswith("reader notebook 1") for cmd in commands)
    assert not any("--mode" in cmd for cmd in commands)
    assert not any("--edit" in cmd for cmd in commands)


def test_next_steps_prefers_config_notebook_template() -> None:
    cfg = _base_config()
    cfg["notebooks"] = {"specs": [{"id": "basic", "template": "notebook/basic"}]}
    spec = ReaderSpec.model_validate(cfg)
    steps = build_next_steps(build_decl(spec), job_label="1")
    notes = [desc for _, desc in steps]
    assert any("template notebook/basic" in desc for desc in notes)


def test_next_steps_uses_cytometry_notebook_for_cytometry_pipeline() -> None:
    cfg = base_reader_config(
        experiment_id="exp",
        pipeline_steps=[{"id": "ingest_cyto", "plugin": "ingest/flow_cytometer"}],
        plot_specs=[],
        export_specs=[],
    )
    spec = ReaderSpec.model_validate(cfg)
    notes = [desc for _, desc in build_next_steps(build_decl(spec), job_label="1")]
    assert any("template notebook/cytometry" in desc for desc in notes)


class _PluginCfg(PluginConfig):
    pass


class _PluginDummy(Plugin):
    ConfigModel = _PluginCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {"artifacts": file_bundle_output("artifacts")}

    def run(self, ctx, inputs, cfg):
        raise AssertionError("not used")


def test_plugins_command_shows_workbench_semantics(monkeypatch) -> None:
    reg = Registry(contracts=builtin_contract_catalog())
    reg.register(
        build_plugin_asset(
            plugin_id="plot/dummy",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="test_plot",
                summary="Synthetic plot plugin for CLI tests.",
            ),
            plugin_cls=_PluginDummy,
        )
    )
    test_console = Console(width=100, record=True, theme=cli.THEME, force_terminal=True)
    monkeypatch.setattr(cli, "console", test_console)
    monkeypatch.setattr(
        cli,
        "builtin_runtime",
        lambda: ReaderRuntime(contracts=builtin_contract_catalog(), plugins=reg, assets=AssetCatalog([])),
    )

    cli.plugins(category=None, domain=None, family=None)

    output = test_console.export_text()
    assert "plate_reader" in output
    assert "test_plot" in output
    assert "Synthetic plot plugin" in output
    assert "for CLI tests." in output


def test_recipes_command_filters_by_family() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["recipes", "--family", "plot_set"])
    assert result.exit_code == 0
    assert "plot_set" in result.output
    assert "synergy_h1" not in result.output


def test_notebook_list_templates_command_shows_semantics() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["notebook", "--list-templates"])
    assert result.exit_code == 0
    assert "Notebook templates" in result.output
    assert "notebook/cytometry" in result.output
    assert "cytometry" in result.output
    assert "record_explorer" in result.output


def test_demo_command_lists_expected_workbench_lifecycle() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["demo"])
    assert result.exit_code == 0
    assert "reader ls" in result.output
    assert "reader validate 1" in result.output
    assert "reader run 1" in result.output
    assert "reader records 1" in result.output
    assert "reader notebook 1" in result.output
