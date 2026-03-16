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
from reader.protocols import builtin_protocol_catalog
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
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_parameters={"fold_change": {"report_times": [14.0]}},
        protocol_analysis={
            "crosstalk_pairs": {"enabled": True, "export": True},
        },
        protocol_deliverables={
            "plots": {"profile": "none", "include": ["time_series"]},
            "exports": {"include": ["crosstalk_pairs_csv"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
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


def test_next_steps_commands_are_clean(tmp_path: Path) -> None:
    spec = ReaderSpec.load(write_config(tmp_path / "config.yaml", _base_config()))
    steps = build_next_steps(build_decl(spec), job_label="1")
    commands = [cmd for cmd, _ in steps]
    assert any(cmd.startswith("reader records 1") for cmd in commands)
    assert any(cmd.startswith("reader plot 1") for cmd in commands)
    assert any("reader export 1" in cmd for cmd in commands)
    assert any(cmd.startswith("reader notebook 1") for cmd in commands)
    assert not any("--mode" in cmd for cmd in commands)
    assert not any("--edit" in cmd for cmd in commands)


def test_next_steps_prefers_config_notebook_template(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["protocol"]["deliverables"]["notebook"] = {"template": "notebook/basic"}
    spec = ReaderSpec.load(write_config(tmp_path / "config.yaml", cfg))
    steps = build_next_steps(build_decl(spec), job_label="1")
    notes = [desc for _, desc in steps]
    assert any("template notebook/basic" in desc for desc in notes)


def test_next_steps_uses_protocol_default_notebook(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp",
        protocol_id="cytometry/flow_panel",
        protocol_parameters={
            "ingest": {"auto_roots": ["./inputs"]},
            "metadata": {"require_columns": ["design_id", "treatment"]},
        },
        resources={"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}},
    )
    tmp_cfg = write_config(tmp_path, cfg)
    spec = ReaderSpec.load(tmp_cfg)
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
        lambda: ReaderRuntime(
            contracts=builtin_contract_catalog(),
            protocols=builtin_protocol_catalog(),
            plugins=reg,
            assets=AssetCatalog([]),
        ),
    )

    cli.plugins(category=None, domain=None, family=None)

    output = test_console.export_text()
    assert "plate_reader" in output
    assert "test_plot" in output
    assert "Synthetic plot plugin" in output
    assert "for CLI tests." in output


def test_protocols_command_filters_by_family() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["protocols", "--family", "screen_analysis"])
    assert result.exit_code == 0
    assert "screen_analysis" in result.output
    assert "plate_reader" in result.output
    assert "cytometry/flow_panel" not in result.output


def test_protocols_command_lists_builtin_protocols() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["protocols", "plate_reader/dual_reporter_screen"])
    assert result.exit_code == 0
    assert "Protocol:" in result.output
    assert "plate_reader/dual_reporter_screen" in result.output
    assert "Dual-reporter screen protocol" in result.output
    assert "notebook/eda" in result.output


def test_notebook_list_templates_command_shows_semantics() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["notebook", "--list-templates"])
    assert result.exit_code == 0
    assert "Notebook templates" in result.output
    assert "notebook/cytometry" in result.output
    assert "cytometry" in result.output
    assert "record_explorer" in result.output


def test_notebook_list_templates_respects_protocol(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp",
        protocol_id="logic/sfxi_screen",
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    cfg_path = write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["notebook", str(cfg_path), "--list-templates"])
    assert result.exit_code == 0
    assert "Notebook templates: logic/sfxi_screen" in result.output
    assert "SFXI vec8" in result.output
    assert "yes" in result.output
    assert "notebook/cytometry" not in result.output


def test_demo_command_lists_expected_workbench_lifecycle() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["demo"])
    assert result.exit_code == 0
    assert "reader ls" in result.output
    assert "reader validate 1" in result.output
    assert "reader run 1" in result.output
    assert "reader records 1" in result.output
    assert "reader notebook 1" in result.output
