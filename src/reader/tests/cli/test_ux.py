"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/cli/test_ux.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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
from reader.workbench.records import RecordStore
from reader.workbench.registry import Plugin, PluginConfig, Registry


def _tidy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["OD600"],
            "value": [1.0],
        }
    )


def _base_config() -> dict:
    return base_reader_config(
        experiment_id="exp",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}},
        protocol_analysis={
            "crosstalk_pairs": {"enabled": True, "export": True},
        },
        protocol_outputs={
            "plots": {"profile": "none", "include": ["raw_kinetics"]},
            "exports": {"include": ["crosstalk_pairs_table"]},
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


def test_ls_details_shows_protocol_and_output_counts(monkeypatch, tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    exp_dir = exp_root / "2025" / "real_exp"
    exp_dir.mkdir(parents=True)
    write_config(exp_dir / "config.yaml", _base_config())

    outputs = exp_dir / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    plots_dir = outputs / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "trace.pdf").write_text("plot", encoding="utf-8")

    test_console = Console(width=120, record=True, theme=cli.THEME, force_terminal=True)
    monkeypatch.setattr(cli, "console", test_console)
    cli.ls(root=str(exp_root), include_scaffolds=False, details=True)

    output = test_console.export_text()
    assert "Protocol" in output
    assert "plate_reader/dual_repo" in output
    assert "Selected" in output
    assert "Generated" in output
    assert "1 rec" in output


def test_ls_json_surfaces_counts_and_config_errors(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    good_dir = exp_root / "2025" / "good_exp"
    bad_dir = exp_root / "2025" / "broken_exp"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)
    write_config(good_dir / "config.yaml", _base_config())
    (bad_dir / "config.yaml").write_text("schema: reader/v7\nexperiment:\n  id: broken\n", encoding="utf-8")

    outputs = good_dir / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["ls", "--root", str(exp_root), "--details", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["count"] == 2
    assert payload["summary"]["status"] == {"config_error": 1, "ok": 1}
    assert payload["summary"]["protocols"] == {"plate_reader/dual_reporter_screen": 1}
    assert payload["summary"]["with_outputs"] == 1
    assert payload["summary"]["without_outputs"] == 1
    by_name = {item["name"]: item for item in payload["experiments"]}
    assert by_name["good_exp"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert by_name["good_exp"]["generated"]["records"] == 1
    assert by_name["good_exp"]["selected"]["plots"]["count"] == 1
    assert by_name["good_exp"]["selected"]["exports"]["ids"] == ["crosstalk_pairs_table"]
    assert by_name["good_exp"]["selected"]["plot_profile"] == "none"
    assert by_name["good_exp"]["status"] == "ok"
    assert by_name["broken_exp"]["status"] == "config_error"
    assert "protocol" in by_name["broken_exp"]["error"]


def test_ls_can_filter_by_protocol_and_status(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    good_dir = exp_root / "2025" / "good_plate"
    cyto_dir = exp_root / "2025" / "good_cytometry"
    broken_dir = exp_root / "2025" / "broken_exp"
    good_dir.mkdir(parents=True)
    cyto_dir.mkdir(parents=True)
    broken_dir.mkdir(parents=True)
    write_config(good_dir / "config.yaml", _base_config())
    write_config(
        cyto_dir / "config.yaml",
        base_reader_config(
            experiment_id="good_cytometry",
            protocol_id="cytometry/flow_panel",
            protocol_inputs={
                "ingest": {"auto_roots": ["./inputs"]},
                "metadata": {"require_columns": ["design_id", "treatment"]},
            },
            resources={"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}},
        ),
    )
    (broken_dir / "config.yaml").write_text("schema: reader/v7\nexperiment:\n  id: broken\n", encoding="utf-8")

    runner = CliRunner()
    by_protocol = runner.invoke(
        cli.app,
        [
            "ls",
            "--root",
            str(exp_root),
            "--details",
            "--protocol",
            "plate_reader/dual_reporter_screen",
            "--format",
            "json",
        ],
    )
    assert by_protocol.exit_code == 0
    protocol_payload = json.loads(by_protocol.output)
    assert protocol_payload["count"] == 1
    assert protocol_payload["filters"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert protocol_payload["experiments"][0]["name"] == "good_plate"

    by_status = runner.invoke(
        cli.app,
        ["ls", "--root", str(exp_root), "--details", "--status", "config_error", "--format", "json"],
    )
    assert by_status.exit_code == 0
    status_payload = json.loads(by_status.output)
    assert status_payload["count"] == 1
    assert status_payload["filters"]["status"] == "config_error"
    assert status_payload["experiments"][0]["name"] == "broken_exp"

    no_matches = runner.invoke(
        cli.app,
        ["ls", "--root", str(exp_root), "--details", "--protocol", "logic/sfxi_screen", "--format", "json"],
    )
    assert no_matches.exit_code == 0
    empty_payload = json.loads(no_matches.output)
    assert empty_payload["count"] == 0
    assert empty_payload["filters"]["protocol"] == "logic/sfxi_screen"
    assert empty_payload["experiments"] == []


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
    cfg["protocol"]["outputs"]["notebook"] = {"template": "notebook/basic"}
    spec = ReaderSpec.load(write_config(tmp_path / "config.yaml", cfg))
    steps = build_next_steps(build_decl(spec), job_label="1")
    notes = [desc for _, desc in steps]
    assert any("template notebook/basic" in desc for desc in notes)


def test_next_steps_uses_protocol_default_notebook(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp",
        protocol_id="cytometry/flow_panel",
        protocol_inputs={
            "ingest": {"auto_roots": ["./inputs"]},
            "metadata": {"require_columns": ["design_id", "treatment"]},
        },
        resources={"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}},
    )
    tmp_cfg = write_config(tmp_path, cfg)
    spec = ReaderSpec.load(tmp_cfg)
    notes = [desc for _, desc in build_next_steps(build_decl(spec), job_label="1")]
    assert any("template notebook/cytometry" in desc for desc in notes)


def test_steps_json_surfaces_pipeline_bindings(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["steps", str(cfg), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["semantic_program"]["metrics"][0]["id"] == "OD"
    assert payload["semantic_program"]["metrics"][0]["execution"]["status"] == "compiled"
    assert payload["count"] >= 1
    first = payload["pipeline"][0]
    assert first["stage"] == "ingest"
    assert first["semantics"]["category"] == "ingest"
    assert first["writes"][0]["display"] == "ingest/df"


def test_explain_json_surfaces_compiled_plan(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["explain", str(cfg), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["semantic_program"]["controls"][0]["execution"]["status"] == "descriptive_only"
    assert payload["plan"]["pipeline_flow"][0] == "ingest"
    assert "sample_map" in payload["plan"]["resources"]
    assert payload["plots"][0]["semantics"]["category"] == "plot"
    assert payload["exports"][0]["semantics"]["category"] == "export"


def test_validate_json_surfaces_preflight_summary(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["validate", str(cfg), "--no-files", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["validation"]["status"] == "ok"
    assert payload["validation"]["counts"]["pipeline"] >= 1
    assert payload["validation"]["files"]["mode"] == "skipped"


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
    result = runner.invoke(cli.app, ["protocols", "plate_reader/dual_reporter_screen"], terminal_width=160)
    assert result.exit_code == 0
    assert "Protocol:" in result.output
    assert "plate_reader/dual_reporter_screen" in result.output
    assert "Dual-reporter screen protocol" in result.output
    assert "notebook/eda" in result.output
    assert "Inputs Surface" in result.output
    assert "ingest.mode" in result.output
    assert "Analysis Surface" in result.output
    assert "Semantic Program" in result.output
    assert "Plot Profiles" in result.output
    assert "Plot Outputs" in result.output
    assert "Export Artifacts" in result.output
    assert "Default Pipeline" in result.output
    assert "Plot Implementations" in result.output


def test_protocols_command_can_render_example_config() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["protocols", "plate_reader/dual_reporter_screen", "--example-config"],
        terminal_width=160,
    )
    assert result.exit_code == 0
    assert "Starter YAML" in result.output
    assert "schema: reader/v7" in result.output
    assert "id: plate_reader/dual_reporter_screen" in result.output
    assert "profile: screen_overview" in result.output


def test_init_command_scaffolds_new_experiment(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / "experiments" / "20260317_new_assay"
    result = runner.invoke(cli.app, ["init", str(target), "--protocol", "plate_reader/dual_reporter_screen"])
    assert result.exit_code == 0
    assert (target / "config.yaml").exists()
    assert (target / "inputs").is_dir()
    assert (target / "notebooks").is_dir()
    spec = ReaderSpec.load(target / "config.yaml")
    assert spec.experiment.id == "20260317_new_assay"
    assert spec.protocol.id == "plate_reader/dual_reporter_screen"


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
    assert "reader inspect 1" in result.output
    assert "reader init" in result.output
    assert "plate_reader/dual_reporter_screen" in result.output
    assert "reader validate 1" in result.output
    assert "reader run 1" in result.output
    assert "reader records 1" in result.output
    assert "reader notebook 1" in result.output


def test_inspect_command_surfaces_pipeline_and_outputs(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path / "config.yaml", _base_config())
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "metadata.xlsx").write_text("xlsx", encoding="utf-8")
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    plot_dir = outputs / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "raw_kinetics.pdf").write_text("plot", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["inspect", str(cfg_path)])
    assert result.exit_code == 0
    assert "Experiment overview" in result.output
    assert "Authoring bindings" in result.output
    assert "Semantic Program" in result.output
    assert "fold_change.report_times" in result.output
    assert "Pipeline chain" in result.output
    assert "Plot outputs" in result.output
    assert "Export artifacts" in result.output
    assert "Generated outputs" in result.output
    assert "Record catalog" in result.output
    assert "raw_kinetics" in result.output
    assert "crosstalk_pairs_table" in result.output


def test_inspect_command_can_emit_json(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path / "config.yaml", _base_config())
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "metadata.xlsx").write_text("xlsx", encoding="utf-8")
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["inspect", str(cfg_path), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["semantic_program"]["metrics"][0]["id"] == "OD"
    assert payload["semantic_program"]["metrics"][0]["execution"]["status"] == "compiled"
    assert payload["authoring"]["inputs"]["fold_change"]["report_times"] == [14.0]
    assert payload["generated"]["records"][0]["record_id"] == "ingest/df"
    assert payload["pipeline"][0]["id"] == "ingest"
    assert payload["plots"][0]["id"] == "raw_kinetics"
    assert payload["pipeline"][0]["writes"][0]["surface"]["minimum"] == "tidy.v1"


def test_plugins_command_can_filter_by_protocol(monkeypatch) -> None:
    test_console = Console(width=160, record=True, theme=cli.THEME, force_terminal=True)
    monkeypatch.setattr(cli, "console", test_console)

    cli.plugins(category="transform", domain=None, family=None, protocol="plate_reader/dual_reporter_screen")

    output = test_console.export_text()
    assert "plate_reader/dual_reporter_screen" in output
    assert "Attach well-position sample maps" in output
    assert "Summarize nearest-time fold-change tables" in output
    assert "validator" not in output


def test_protocols_command_can_emit_json() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["protocols", "plate_reader/dual_reporter_screen", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["semantic_program"]["metrics"][0]["id"] == "OD"
    assert payload["semantic_program"]["metrics"][0]["execution"]["status"] == "compiled"
    assert payload["starter_config"]["schema"] == "reader/v7"
    assert payload["compiled"]["pipeline"][0]["id"] == "ingest"
    assert any(item["id"] == "screen_overview" for item in payload["plot_profiles"])


def test_protocols_command_json_surfaces_compiled_logic_semantic_program() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["protocols", "logic/sfxi_screen", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["protocol"] == "logic/sfxi_screen"
    assert payload["semantic_program"]["controls"][0]["execution"]["status"] == "compiled"
    assert payload["semantic_program"]["controls"][0]["execution"]["step_ids"] == ["sfxi_vec8"]
    assert payload["semantic_program"]["windows"][0]["execution"]["status"] == "compiled"
    assert payload["semantic_program"]["metrics"][0]["execution"]["record_ids"] == ["sfxi_vec8/vec8"]
    assert payload["semantic_program"]["ranking"]["execution"]["status"] == "compiled"


def test_plugins_command_can_emit_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["plugins", "--protocol", "plate_reader/dual_reporter_screen", "--category", "transform", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["count"] >= 1
    assert all(item["category"] == "transform" for item in payload["plugins"])


def test_records_command_can_emit_json_with_history(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path / "config.yaml", _base_config())
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test1",
    )
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test2",
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["records", str(cfg_path), "--all", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["all"] is True
    assert payload["records"][0]["record_id"] == "ingest/df"
    assert payload["records"][0]["revision_count"] == 2
