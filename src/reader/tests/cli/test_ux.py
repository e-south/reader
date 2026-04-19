"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/cli/test_ux.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from reader.contracts import builtin_contract_catalog
from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.runtime import ReaderRuntime
from reader.tests.support import base_reader_config, build_decl, write_config
from reader.workbench import PluginSemantics, cli
from reader.workbench.assets import AssetCatalog, build_plugin_asset
from reader.workbench.config import ReaderSpec
from reader.workbench.engine import build_next_steps
from reader.workbench.ports import dataframe_input, file_bundle_output
from reader.workbench.records import RecordStore
from reader.workbench.registry import Plugin, PluginConfig, Registry


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def _compiled_semantic_program(payload: dict) -> dict:
    return payload["implementation"]["compiled"]["semantic_program"]


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
    monkeypatch.setattr(cli.shared, "console", test_console)
    cli.ls(root=str(exp_root), include_scaffolds=False)

    output = test_console.export_text()
    max_line = max(len(line) for line in output.splitlines()) if output else 0
    assert max_line <= 80


def test_ls_excludes_template_dirs_by_default(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    year_dir = exp_root / "2025" / "real_exp"
    template_dir = exp_root / "template"
    year_dir.mkdir(parents=True)
    template_dir.mkdir(parents=True)
    write_config(year_dir / "config.yaml", _base_config())
    write_config(template_dir / "config.yaml", _base_config())

    runner = CliRunner()
    result = runner.invoke(cli.app, ["ls", "--root", str(exp_root), "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [item["name"] for item in payload["experiments"]] == ["real_exp"]


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


def test_numeric_job_index_rejects_hidden_scaffold_index(monkeypatch, tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    year_dir = exp_root / "2025" / "real_exp"
    template_dir = exp_root / "2025" / "_template_alpha"
    template_dir.mkdir(parents=True)
    year_dir.mkdir(parents=True)
    write_config(template_dir / "config.yaml", _base_config())
    write_config(year_dir / "config.yaml", _base_config())

    monkeypatch.chdir(tmp_path)
    with pytest.raises(typer.BadParameter, match="hidden scaffold/template config"):
        cli._infer_job_path("1")
    assert cli._infer_job_path("2") == (year_dir / "config.yaml").resolve()


def test_ls_preserves_shared_numeric_indexes_when_scaffolds_are_hidden(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    year_dir = exp_root / "2025" / "real_exp"
    template_dir = exp_root / "2025" / "_template_alpha"
    template_dir.mkdir(parents=True)
    year_dir.mkdir(parents=True)
    write_config(template_dir / "config.yaml", _base_config())
    write_config(year_dir / "config.yaml", _base_config())

    runner = CliRunner()
    result = runner.invoke(cli.app, ["ls", "--root", str(exp_root), "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [item["name"] for item in payload["experiments"]] == ["real_exp"]
    assert [item["index"] for item in payload["experiments"]] == [2]


def test_ls_all_includes_template_dirs(monkeypatch, tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    year_dir = exp_root / "2025" / "real_exp"
    template_dir = exp_root / "template"
    year_dir.mkdir(parents=True)
    template_dir.mkdir(parents=True)
    write_config(year_dir / "config.yaml", _base_config())
    write_config(template_dir / "config.yaml", _base_config())

    test_console = Console(width=80, record=True, theme=cli.THEME, force_terminal=True)
    monkeypatch.setattr(cli.shared, "console", test_console)
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
    monkeypatch.setattr(cli.shared, "console", test_console)
    cli.ls(root=str(exp_root), include_scaffolds=False, details=True)

    output = test_console.export_text()
    assert "Protocol" in output
    assert "plate_reader/dual_repo" in output
    assert "Selected" in output
    assert "Generated" in output
    assert "1 rec" in output


def test_ls_readiness_requires_details(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    exp_dir = exp_root / "2025" / "real_exp"
    exp_dir.mkdir(parents=True)
    write_config(exp_dir / "config.yaml", _base_config())

    runner = CliRunner()
    result = runner.invoke(cli.app, ["ls", "--root", str(exp_root), "--readiness"])
    assert result.exit_code != 0
    assert "--readiness requires --details" in _plain(result.output)


def test_ls_json_surfaces_counts_and_config_errors(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    good_dir = exp_root / "2025" / "good_exp"
    bad_dir = exp_root / "2025" / "broken_exp"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)
    write_config(good_dir / "config.yaml", _base_config())
    (bad_dir / "config.yaml").write_text("schema: reader/v7\nexperiment:\n  id: broken\n", encoding="utf-8")
    inputs_dir = good_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "metadata.xlsx").write_text("stub", encoding="utf-8")
    (inputs_dir / "20250101_sensor_panel.xlsx").write_text("stub", encoding="utf-8")

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
        ["ls", "--root", str(exp_root), "--details", "--readiness", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["catalog"]["kind"] == "experiments"
    assert payload["catalog"]["root"] == str(exp_root.resolve())
    assert payload["selection"]["details"] is True
    assert payload["selection"]["readiness"] is True
    assert payload["selection"]["include_scaffolds"] is False
    assert payload["summary"]["experiments"] == 2
    assert payload["summary"]["by_lifecycle"] == {"active": 1, "unknown": 1}
    assert payload["summary"]["by_readiness"] == {"config_error": 1, "records_ready": 1}
    assert payload["summary"]["by_status"] == {"config_error": 1, "ok": 1}
    assert payload["summary"]["by_protocol"] == {"plate_reader/dual_reporter_screen": 1}
    assert payload["summary"]["outputs"] == {"with_outputs": 1, "without_outputs": 1}
    by_name = {item["name"]: item for item in payload["experiments"]}
    assert by_name["good_exp"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert by_name["good_exp"]["generated"]["records"] == 1
    assert by_name["good_exp"]["selected"]["plots"]["count"] == 1
    assert by_name["good_exp"]["selected"]["exports"]["ids"] == ["crosstalk_pairs_table"]
    assert by_name["good_exp"]["selected"]["plot_profile"] == "none"
    assert by_name["good_exp"]["lifecycle"] == "active"
    assert by_name["good_exp"]["status"] == "ok"
    assert by_name["good_exp"]["readiness"]["state"] == "records_ready"
    assert by_name["good_exp"]["readiness"]["capabilities"]["plot"] is True
    assert by_name["broken_exp"]["status"] == "config_error"
    assert by_name["broken_exp"]["readiness"]["state"] == "config_error"
    assert "protocol" in by_name["broken_exp"]["error"]


def test_ls_rejects_missing_root(tmp_path: Path) -> None:
    runner = CliRunner()
    missing_root = tmp_path / "missing"

    result = runner.invoke(cli.app, ["ls", "--root", str(missing_root), "--details", "--format", "json"])

    assert result.exit_code != 0
    assert "Experiments root not found" in result.output


def test_ls_json_surfaces_legacy_outputs_without_record_catalog(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    exp_dir = exp_root / "2025" / "legacy_exp"
    exp_dir.mkdir(parents=True)
    write_config(exp_dir / "config.yaml", _base_config())
    inputs_dir = exp_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "metadata.xlsx").write_text("stub", encoding="utf-8")
    (inputs_dir / "20250101_sensor_panel.xlsx").write_text("stub", encoding="utf-8")
    plots_dir = exp_dir / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "trace.pdf").write_text("plot", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["ls", "--root", str(exp_root), "--details", "--readiness", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["by_readiness"] == {"legacy_outputs_present": 1}
    entry = payload["experiments"][0]
    assert entry["readiness"]["state"] == "legacy_outputs_present"
    assert entry["readiness"]["records"]["catalog"] is False
    assert entry["readiness"]["records"]["legacy_outputs_present"] is True


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
    assert protocol_payload["summary"]["experiments"] == 1
    assert protocol_payload["selection"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert protocol_payload["experiments"][0]["name"] == "good_plate"

    by_status = runner.invoke(
        cli.app,
        ["ls", "--root", str(exp_root), "--details", "--status", "config_error", "--format", "json"],
    )
    assert by_status.exit_code == 0
    status_payload = json.loads(by_status.output)
    assert status_payload["summary"]["experiments"] == 1
    assert status_payload["selection"]["status"] == "config_error"
    assert status_payload["experiments"][0]["name"] == "broken_exp"

    no_matches = runner.invoke(
        cli.app,
        ["ls", "--root", str(exp_root), "--details", "--protocol", "logic/sfxi_screen", "--format", "json"],
    )
    assert no_matches.exit_code == 0
    empty_payload = json.loads(no_matches.output)
    assert empty_payload["summary"]["experiments"] == 0
    assert empty_payload["selection"]["protocol"] == "logic/sfxi_screen"
    assert empty_payload["experiments"] == []


def test_ls_can_filter_by_lifecycle_and_surface_draft_readiness(tmp_path: Path) -> None:
    exp_root = tmp_path / "experiments"
    active_dir = exp_root / "2025" / "active_exp"
    draft_dir = exp_root / "2025" / "draft_exp"
    active_dir.mkdir(parents=True)
    draft_dir.mkdir(parents=True)
    write_config(active_dir / "config.yaml", _base_config())
    write_config(draft_dir / "config.yaml", {**_base_config(), "experiment": {"id": "draft_exp", "lifecycle": "draft"}})

    runner = CliRunner()
    by_lifecycle = runner.invoke(
        cli.app,
        ["ls", "--root", str(exp_root), "--details", "--readiness", "--lifecycle", "draft", "--format", "json"],
    )
    assert by_lifecycle.exit_code == 0
    payload = json.loads(by_lifecycle.output)
    assert payload["summary"]["experiments"] == 1
    assert payload["selection"]["lifecycle"] == "draft"
    assert payload["summary"]["by_lifecycle"] == {"draft": 1}
    assert payload["summary"]["by_readiness"] == {"draft": 1}
    assert payload["experiments"][0]["name"] == "draft_exp"
    assert payload["experiments"][0]["lifecycle"] == "draft"
    assert payload["experiments"][0]["readiness"]["state"] == "draft"
    assert payload["experiments"][0]["readiness"]["capabilities"]["run"] is False


def test_next_steps_commands_are_clean(tmp_path: Path) -> None:
    spec = ReaderSpec.load(write_config(tmp_path / "config.yaml", _base_config()))
    steps = build_next_steps(build_decl(spec), job_label="1")
    commands = [cmd for cmd, _ in steps]
    assert any(cmd.startswith("uv run reader records 1") for cmd in commands)
    assert any(cmd.startswith("uv run reader plot 1") for cmd in commands)
    assert any("uv run reader export 1" in cmd for cmd in commands)
    assert any(cmd.startswith("uv run reader notebook 1") for cmd in commands)
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
    compiled_program = _compiled_semantic_program(payload)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["authoring"]["inputs"]["fold_change"]["report_times"] == [14.0]
    assert payload["semantics"]["program"]["metrics"][0]["id"] == "OD"
    assert payload["semantics"]["program"]["active_profile"] == "yfp_cfp_crosstalk"
    assert compiled_program["metrics"][0]["execution"]["status"] == "compiled"
    assert compiled_program["summary"]["compiled"] >= 1
    assert compiled_program["summary"]["descriptive_only"] == 0
    assert compiled_program["active_profile"] == "yfp_cfp_crosstalk"
    assert compiled_program["ranking"]["execution"]["status"] == "compiled"
    assert payload["implementation"]["plan"]["pipeline_count"] >= 1
    assert payload["implementation"]["plan"]["plots"] == []
    assert payload["implementation"]["compiled"]["plots"] == []
    assert payload["implementation"]["compiled"]["exports"] == []
    first = payload["implementation"]["compiled"]["pipeline"][0]
    assert first["stage"] == "ingest"
    assert first["semantics"]["category"] == "ingest"
    assert first["writes"][0]["display"] == "ingest/df"
    assert "semantic_program" not in payload
    assert "count" not in payload
    assert "pipeline" not in payload


def test_config_json_surfaces_authoring_semantics_and_implementation(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["config", str(cfg), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    compiled_program = _compiled_semantic_program(payload)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["authoring"]["schema"] == "reader/v7"
    assert payload["authoring"]["protocol"]["id"] == "plate_reader/dual_reporter_screen"
    assert payload["semantics"]["program"]["metrics"][0]["id"] == "OD"
    assert payload["semantics"]["program"]["active_profile"] == "yfp_cfp_crosstalk"
    assert payload["semantics"]["program"]["controls"] == []
    assert payload["semantics"]["program"]["windows"] == []
    assert payload["semantics"]["program"]["ranking"]["primary_metric"] == "log2FC"
    assert compiled_program["metrics"][0]["execution"]["status"] == "compiled"
    assert compiled_program["active_profile"] == "yfp_cfp_crosstalk"
    assert payload["implementation"]["plan"]["pipeline_flow"][0] == "ingest"
    assert payload["implementation"]["compiled"]["pipeline"][0]["id"] == "ingest"
    assert payload["implementation"]["compiled"]["plots"][0]["id"] == "raw_kinetics"
    assert "compiled" not in payload
    assert "protocol" not in payload


def test_explain_json_surfaces_compiled_plan(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["explain", str(cfg), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    compiled_program = _compiled_semantic_program(payload)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["authoring"]["inputs"]["fold_change"]["report_times"] == [14.0]
    assert payload["semantics"]["program"]["active_profile"] == "yfp_cfp_crosstalk"
    assert payload["semantics"]["program"]["controls"] == []
    assert payload["semantics"]["program"]["windows"] == []
    assert payload["semantics"]["program"]["summary"]["total"] >= 1
    assert compiled_program["ranking"]["execution"]["status"] == "compiled"
    assert payload["implementation"]["plan"]["pipeline_flow"][0] == "ingest"
    assert "sample_map" in payload["implementation"]["plan"]["resources"]
    assert payload["implementation"]["compiled"]["plots"][0]["semantics"]["category"] == "plot"
    assert payload["implementation"]["compiled"]["exports"][0]["semantics"]["category"] == "export"
    assert "semantic_program" not in payload
    assert "plan" not in payload
    assert "pipeline" not in payload


def test_validate_json_surfaces_preflight_summary(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(cli.app, ["validate", str(cfg), "--no-files", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["selection"]["check_files"] is False
    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["counts"]["pipeline"] >= 1
    assert payload["validation"]["files"]["mode"] == "skipped"
    assert payload["validation"]["files"]["declared"]["file_inputs"] >= 1
    assert payload["validation"]["files"]["declared"]["auto_roots"] == 0
    assert payload["validation"]["dependencies"]["checked"] is False
    assert payload["validation"]["errors"] == []
    assert "protocol" not in payload["validation"]


def test_validate_json_surfaces_file_check_selection(monkeypatch, tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "metadata.xlsx").write_text("stub", encoding="utf-8")
    (inputs_dir / "20250101_sensor_panel.xlsx").write_text("stub", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["validate", str(cfg), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["selection"]["check_files"] is True
    assert payload["summary"]["status"] == "ok"
    assert payload["validation"]["files"]["checked"] is True
    assert payload["validation"]["dependencies"]["checked"] is True
    assert payload["validation"]["errors"] == []


def test_validate_json_surfaces_runtime_readiness_errors(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "metadata.xlsx").write_text("stub", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["validate", str(cfg), "--format", "json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["summary"]["status"] == "error"
    assert payload["validation"]["files"]["mode"] == "error"
    assert payload["validation"]["dependencies"]["summary"] == "ok"
    assert any("No raw .xlsx files discovered" in item for item in payload["validation"]["errors"])


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
    monkeypatch.setattr(cli.shared, "console", test_console)
    monkeypatch.setattr(
        "reader.runtime.builtin_runtime",
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
    assert "Dual-reporter plate-reader panel protocol" in result.output
    assert "notebook/eda" in result.output
    assert "Inputs" in result.output
    assert "ingest.mode" in result.output
    assert "Analysis" in result.output
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
    assert "channels:" not in result.output
    assert "target:" not in result.output


def test_single_reporter_protocol_example_config_surfaces_semantic_channels() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["protocols", "plate_reader/single_reporter_screen", "--example-config"],
        terminal_width=160,
    )

    assert result.exit_code == 0
    assert "id: plate_reader/single_reporter_screen" in result.output
    assert "reporter_channel: RFP" in result.output
    assert "normalizer_channel: OD600" in result.output
    assert "channels:" not in result.output
    assert "target:" not in result.output


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
    assert "notebook/retron_s" in result.output
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


def test_notebook_list_templates_marks_resolved_scaffold_template(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp",
        protocol_id="plate_reader/retron_sponge_screen",
        protocol_analysis={
            "semantic_metrics": {
                "relevant_stress_map": {"sulAp": "100 nM ciprofloxacin"},
                "sensor_target_map": {"sulAp": ["LexA"]},
            }
        },
        protocol_outputs={"notebook": {"template": "notebook/basic"}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    cfg_path = write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["notebook", str(cfg_path), "--list-templates"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "Scaffold" in result.output
    basic_line = next(line for line in result.output.splitlines() if line.strip().startswith("│ │ notebook/basic"))
    retron_line = next(
        line for line in result.output.splitlines() if line.strip().startswith("│ │ notebook/retron_sponge")
    )
    assert "yes" in basic_line
    assert "yes" not in retron_line


def test_demo_command_lists_expected_workbench_lifecycle() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["demo"])
    assert result.exit_code == 0
    assert "uv run reader ls" in result.output
    assert "--details" in result.output
    assert "--readiness" in result.output
    assert "uv run reader inspect" in result.output
    assert "<experiment>" in result.output
    assert "config.ya" in result.output
    assert "reader init" in result.output
    assert "<protocol-id>" in result.output
    assert "uv run reader validate" in result.output
    assert "uv run reader run" in result.output
    assert "uv run reader records" in result.output
    assert "uv run reader notebook" in result.output


def test_inspect_command_surfaces_pipeline_and_outputs(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path / "config.yaml", _base_config())
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "metadata.xlsx").write_text("xlsx", encoding="utf-8")
    (inputs_dir / "20250101_sensor_panel.xlsx").write_text("xlsx", encoding="utf-8")
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
    assert "Readiness" in result.output
    assert "records ready" in result.output
    assert "Config values" in result.output
    assert "Semantic Program" in result.output
    assert "fold_change.report_times" in result.output
    assert "Pipeline chain" in result.output
    assert "Plot outputs" in result.output
    assert "Exports" in result.output
    assert "Generated outputs" in result.output
    assert "Records" in result.output
    assert "raw_kinetics" in result.output
    assert "crosstalk_pairs_table" in result.output


def test_inspect_command_can_emit_json(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path / "config.yaml", _base_config())
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "metadata.xlsx").write_text("xlsx", encoding="utf-8")
    (inputs_dir / "20250101_sensor_panel.xlsx").write_text("xlsx", encoding="utf-8")
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
    compiled_program = _compiled_semantic_program(payload)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["experiment"]["lifecycle"] == "active"
    assert payload["semantics"]["program"]["metrics"][0]["id"] == "OD"
    assert payload["semantics"]["program"]["active_profile"] == "yfp_cfp_crosstalk"
    assert compiled_program["metrics"][0]["execution"]["status"] == "compiled"
    assert compiled_program["summary"]["descriptive_only"] == 0
    assert compiled_program["active_profile"] == "yfp_cfp_crosstalk"
    assert compiled_program["ranking"]["execution"]["status"] == "compiled"
    assert payload["authoring"]["inputs"]["fold_change"]["report_times"] == [14.0]
    assert "sample_map" in payload["implementation"]["plan"]["resources"]
    assert payload["implementation"]["inputs"]["counts"]["files"] == 2
    assert payload["implementation"]["readiness"]["state"] == "records_ready"
    assert payload["implementation"]["readiness"]["capabilities"]["run"] is True
    assert payload["implementation"]["readiness"]["capabilities"]["plot"] is True
    assert payload["implementation"]["readiness"]["records"]["catalog"] is True
    assert payload["implementation"]["generated"]["records"][0]["record_id"] == "ingest/df"
    assert payload["implementation"]["compiled"]["pipeline"][0]["id"] == "ingest"
    assert payload["implementation"]["compiled"]["plots"][0]["id"] == "raw_kinetics"
    assert payload["implementation"]["compiled"]["pipeline"][0]["writes"][0]["surface"]["minimum"] == "tidy.v1"
    assert "semantic_program" not in payload
    assert "generated" not in payload
    assert "pipeline" not in payload


def test_plugins_command_can_filter_by_protocol(monkeypatch) -> None:
    test_console = Console(width=160, record=True, theme=cli.THEME, force_terminal=True)
    monkeypatch.setattr(cli.shared, "console", test_console)

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
    compiled_program = _compiled_semantic_program(payload)
    metrics = {item["id"]: item for item in payload["semantics"]["program"]["metrics"]}
    assert payload["protocol"] == "plate_reader/dual_reporter_screen"
    assert metrics["Ratio"]["formula"] == "YFP / CFP"
    assert metrics["Ratio"]["value_space"] == "linear_ratio"
    assert metrics["Ratio"]["unit"] == "ratio"
    assert metrics["Ratio"]["comparable_group"] == "primary_ratio_linear"
    assert payload["semantics"]["program"]["active_profile"] == "yfp_cfp_fold_change"
    assert payload["semantics"]["program"]["controls"] == []
    assert payload["semantics"]["program"]["windows"] == []
    assert payload["semantics"]["program"]["ranking"] is None
    compiled_metrics = {item["id"]: item for item in compiled_program["metrics"]}
    assert compiled_metrics["OD"]["execution"]["status"] == "compiled"
    assert compiled_metrics["Ratio"]["execution"]["step_ids"] == ["ratio_yfp_cfp"]
    assert compiled_program["summary"]["descriptive_only"] == 0
    assert compiled_metrics["FC"]["execution"]["step_ids"] == ["fold_change__yfp_over_cfp"]
    assert compiled_metrics["log2FC"]["execution"]["record_ids"] == ["fold_change__yfp_over_cfp/table"]
    assert payload["authoring"]["starter_config"]["schema"] == "reader/v7"
    assert payload["implementation"]["compiled"]["pipeline"][0]["id"] == "ingest"
    assert any(item["id"] == "screen_overview" for item in payload["authoring"]["outputs"]["plot_profiles"])
    assert payload["implementation"]["defaults"][0]["parameters"]["mode"]["source"] == "protocol.inputs.ingest.mode"
    assert "plugin_defaults" not in payload
    assert "semantic_program" not in payload


def test_protocols_command_json_surfaces_compiled_logic_semantic_program() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["protocols", "logic/sfxi_screen", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    compiled_program = _compiled_semantic_program(payload)
    assert payload["protocol"] == "logic/sfxi_screen"
    assert payload["semantics"]["program"]["summary"]["total"] == 4
    assert compiled_program["summary"]["compiled"] == 4
    assert compiled_program["summary"]["descriptive_only"] == 0
    assert compiled_program["controls"][0]["execution"]["status"] == "compiled"
    assert compiled_program["controls"][0]["execution"]["step_ids"] == ["sfxi_vec8"]
    assert compiled_program["windows"][0]["execution"]["status"] == "compiled"
    assert compiled_program["metrics"][0]["execution"]["record_ids"] == ["sfxi_vec8/vec8"]
    assert compiled_program["ranking"]["execution"]["status"] == "compiled"


def test_protocols_command_json_surfaces_retron_sponge_semantics() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["protocols", "plate_reader/retron_sponge_screen", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    compiled_program = _compiled_semantic_program(payload)
    program = payload["semantics"]["program"]
    metrics = {item["id"]: item for item in program["metrics"]}
    compiled_metrics = {item["id"]: item for item in compiled_program["metrics"]}
    figure_ids = {item["id"] for item in payload["authoring"]["outputs"]["figures"]}
    plot_profile_ids = {item["id"] for item in payload["authoring"]["outputs"]["plot_profiles"]}
    assert payload["protocol"] == "plate_reader/retron_sponge_screen"
    assert payload["authoring"]["outputs"]["default_plot_profile"] == "screen_overview"
    assert (
        payload["authoring"]["outputs"]["notebook_policy"]["summary"]
        == "Retron sponge screens default to the protocol-specific review notebook and keep the generic record explorers available as fallbacks."
    )
    assert figure_ids == {
        "raw_kinetics",
        "support_kinetics",
        "control_burden_panel",
        "baseline_shifted_kinetics",
        "matched_control_kinetics",
        "induced_effect_kinetics",
        "absolute_effect_kinetics",
        "control_anchored_decomposition",
        "interaction_summary",
        "library_heatmaps",
        "stress_modulation_scores",
        "pareto_ranking",
    }
    assert plot_profile_ids == {"screen_overview", "kinetics_qc", "analysis_review"}
    assert "ratio_heatmap" not in figure_ids
    assert "support_heatmap" not in figure_ids
    assert {item["id"] for item in payload["authoring"]["outputs"]["artifacts"]} == {
        "semantic_trace_table",
        "semantic_summary_table",
    }
    assert program["active_profile"] == "yfp_cfp"
    assert metrics["R"]["formula"] == "log2(YFP / CFP)"
    assert metrics["R"]["value_space"] == "log2_ratio"
    assert metrics["R"]["unit"] == "log2_ratio"
    assert metrics["R"]["comparable_group"] == "primary_ratio_log2"
    assert compiled_metrics["R"]["execution"]["status"] == "compiled"
    assert compiled_metrics["R"]["execution"]["record_ids"] == ["semantic_metrics/trace"]
    assert compiled_metrics["D_AUC"]["execution"]["status"] == "compiled"
    assert compiled_metrics["D_AUC"]["execution"]["record_ids"] == ["semantic_metrics/summary"]
    assert compiled_metrics["D_abs_AUC"]["execution"]["record_ids"] == ["semantic_metrics/summary"]
    assert compiled_metrics["D_growth_AUC"]["execution"]["record_ids"] == ["semantic_metrics/summary"]
    assert compiled_program["controls"][0]["execution"]["status"] == "compiled"
    assert compiled_program["windows"][0]["execution"]["step_ids"] == ["semantic_metrics"]
    assert program["ranking"]["primary_metric"] == "O_abs_AUC"
    assert compiled_program["ranking"]["execution"]["record_ids"] == ["semantic_metrics/summary"]
    assert payload["implementation"]["compiled"]["pipeline"][-1]["id"] == "semantic_metrics"


def test_inspect_json_surfaces_active_single_reporter_semantic_profile(tmp_path: Path) -> None:
    cfg_path = write_config(
        tmp_path / "config.yaml",
        base_reader_config(
            experiment_id="exp_rfp",
            protocol_id="plate_reader/single_reporter_screen",
            protocol_inputs={"fold_change": {"report_times": [14.0]}},
            protocol_analysis={
                "reporter_channel": "mCherry",
                "normalizer_channel": "OD700",
                "include_fold_change": False,
            },
            protocol_outputs={"plots": {"profile": "none", "include": ["raw_kinetics"]}},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        ),
    )
    runner = CliRunner()
    result = runner.invoke(cli.app, ["inspect", str(cfg_path), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    compiled_program = _compiled_semantic_program(payload)
    program = payload["semantics"]["program"]
    metrics = {item["id"]: item for item in program["metrics"]}
    compiled_metrics = {item["id"]: item for item in compiled_program["metrics"]}
    assert program["active_profile"] == "single_reporter_raw"
    assert {item["id"] for item in program["profiles"]} == {
        "single_reporter_raw",
        "single_reporter_fold_change",
    }
    assert set(metrics) == {"Normalizer", "Reporter", "Reporter_Normalizer"}
    assert metrics["Normalizer"]["formula"] == "configured_normalizer_channel"
    assert metrics["Reporter_Normalizer"]["formula"] == "configured_reporter_channel / configured_normalizer_channel"
    assert compiled_metrics["Reporter"]["execution"]["status"] == "compiled"
    assert (
        compiled_metrics["Normalizer"]["execution"]["note"]
        == "Raw OD700 values are materialized on the ingest dataframe."
    )
    assert compiled_metrics["Reporter_Normalizer"]["execution"]["step_ids"] == ["ratio_reporter_normalizer"]
    assert program["controls"] == []
    assert program["windows"] == []
    assert program["ranking"] is None
    assert program["summary"] == {
        "total": 3,
        "by_kind": {
            "control_rule": 0,
            "window": 0,
            "metric": 3,
            "ranking": 0,
        },
    }
    assert compiled_program["summary"]["compiled"] == 3
    assert compiled_program["summary"]["descriptive_only"] == 0


def test_inspect_json_surfaces_active_single_reporter_retron_sponge_profile(tmp_path: Path) -> None:
    cfg_path = write_config(
        tmp_path / "config.yaml",
        base_reader_config(
            experiment_id="exp_rfp_sponge",
            protocol_id="plate_reader/retron_sponge_screen",
            protocol_analysis={
                "measurement": "single_reporter",
                "reporter_channel": "mCherry",
                "growth_channel": "OD700",
                "include_fold_change": False,
                "semantic_metrics": {
                    "relevant_stress_map": {"sulAp": "100 nM ciprofloxacin"},
                    "sensor_target_map": {"sulAp": ["LexA"]},
                },
            },
            protocol_outputs={"plots": {"profile": "none", "include": ["raw_kinetics"]}},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        ),
    )
    runner = CliRunner()
    result = runner.invoke(cli.app, ["inspect", str(cfg_path), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    compiled_program = _compiled_semantic_program(payload)
    program = payload["semantics"]["program"]
    metrics = {item["id"]: item for item in program["metrics"]}
    compiled_metrics = {item["id"]: item for item in compiled_program["metrics"]}

    assert program["active_profile"] == "single_reporter"
    assert {item["id"] for item in program["profiles"]} == {"yfp_cfp", "single_reporter"}
    assert {
        "OD",
        "Reporter",
        "R",
        "Reporter_OD",
        "R_pre",
        "P_pre",
        "B",
        "C",
        "D",
        "D_abs",
        "O",
        "O_abs",
        "S_abs_AUC",
    } <= set(metrics)
    assert "YFP" not in metrics
    assert "CFP" not in metrics
    assert metrics["OD"]["formula"] == "configured_growth_channel"
    assert metrics["OD"]["summary"] == "Raw configured growth-proxy trace."
    assert metrics["R"]["formula"] == "log2(configured_reporter_channel / configured_growth_channel)"
    assert metrics["mu"]["formula"] == "d(log(configured_growth_channel)) / dt"
    assert metrics["R"]["value_space"] == "log2_ratio"
    assert metrics["Reporter_OD"]["formula"] == "configured_reporter_channel / configured_growth_channel"
    assert compiled_metrics["Reporter"]["execution"]["status"] == "compiled"
    assert compiled_metrics["R"]["execution"]["step_ids"] == ["semantic_metrics"]
    assert compiled_metrics["Reporter_OD"]["execution"]["step_ids"] == ["ratio_reporter_normalizer"]
    assert (
        compiled_metrics["Reporter_OD"]["execution"]["note"]
        == "The mCherry/OD700 support channel is materialized as a ratio step output."
    )
    assert compiled_program["controls"][0]["execution"]["step_ids"] == ["semantic_metrics"]
    assert program["ranking"]["primary_metric"] == "O_abs_AUC"
    assert compiled_program["ranking"]["execution"]["record_ids"] == ["semantic_metrics/summary"]


def test_plate_reader_single_reporter_compiler_derives_channels_from_analysis() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/single_reporter_screen",
            inputs={"fold_change": {"report_times": [14.0]}},
            analysis={"reporter_channel": "mCherry", "normalizer_channel": "OD700"},
        )
    )

    plan = protocol.compile()
    ingest = next(step for step in plan.pipeline if step.id == "ingest")
    fold_change = next(step for step in plan.pipeline if step.id == "fold_change__single_reporter")

    assert ingest.with_["channels"] == ["OD700", "mCherry"]
    assert fold_change.with_["target"] == "mCherry/OD700"


def test_plate_reader_retron_sponge_compiler_derives_dual_reporter_ingest_channels() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="plate_reader/retron_sponge_screen"))

    plan = protocol.compile()
    ingest = next(step for step in plan.pipeline if step.id == "ingest")
    semantic_metrics = next(step for step in plan.pipeline if step.id == "semantic_metrics")
    plot_ids = {step.id for step in plan.plots}
    raw_kinetics = next(step for step in plan.plots if step.id == "raw_kinetics")
    support_kinetics = next(step for step in plan.plots if step.id == "support_kinetics")
    control_burden = next(step for step in plan.plots if step.id == "control_burden_panel")
    absolute_effect = next(step for step in plan.plots if step.id == "absolute_effect_kinetics")
    induced_effect = next(step for step in plan.plots if step.id == "induced_effect_kinetics")
    decomposition = next(step for step in plan.plots if step.id == "control_anchored_decomposition")
    library_heatmaps = next(step for step in plan.plots if step.id == "library_heatmaps")
    pareto_ranking = next(step for step in plan.plots if step.id == "pareto_ranking")
    summary_export = next(step for step in plan.exports if step.id == "semantic_summary_table")
    trace_export = next(step for step in plan.exports if step.id == "semantic_trace_table")

    assert plot_ids == {
        "raw_kinetics",
        "support_kinetics",
        "control_burden_panel",
        "control_anchored_decomposition",
        "absolute_effect_kinetics",
        "induced_effect_kinetics",
        "library_heatmaps",
        "pareto_ranking",
    }
    assert ingest.with_["channels"] == ["OD600", "CFP", "YFP"]
    assert semantic_metrics.with_["measurement_channel"] == "YFP/CFP"
    assert semantic_metrics.reads["df"].record_id == "ratio_yfp_cfp/df"
    assert raw_kinetics.with_["ylabel_map"] == {"OD600": "OD600", "CFP": "CFP", "YFP": "YFP"}
    assert support_kinetics.with_["y"] == ["YFP/OD600", "CFP/OD600"]
    assert support_kinetics.with_["ylabel_map"] == {"YFP/OD600": "YFP/OD600", "CFP/OD600": "CFP/OD600"}
    assert control_burden.with_["metrics"] == ["R", "mu"]
    assert control_burden.with_["metric_label_map"] == {"R": "log2(YFP/CFP)", "mu": "d ln(OD600) / dt"}
    assert control_burden.reads["trace"].record_id == "semantic_metrics/trace"
    assert induced_effect.with_["panel_by"] == "sponge"
    assert absolute_effect.with_["metrics"] == ["D_abs"]
    assert absolute_effect.with_["panel_by"] == "sponge"
    assert decomposition.with_["view"] == "decomposition"
    assert decomposition.reads["summary"].record_id == "semantic_metrics/summary"
    assert decomposition.reads["trace"].record_id == "semantic_metrics/trace"
    assert library_heatmaps.reads["summary"].record_id == "semantic_metrics/summary"
    assert library_heatmaps.reads["trace"].record_id == "semantic_metrics/trace"
    assert pareto_ranking.reads["summary"].record_id == "semantic_metrics/summary"
    assert pareto_ranking.reads["trace"].record_id == "semantic_metrics/trace"
    assert pareto_ranking.with_["metric"] == "S_abs_AUC"
    assert pareto_ranking.with_["burden_metric"] == "D_growth_AUC"
    assert summary_export.reads["df"].record_id == "semantic_metrics/summary"
    assert summary_export.with_["path"] == "retron/semantic_summary.csv"
    assert trace_export.reads["df"].record_id == "semantic_metrics/trace"
    assert trace_export.with_["path"] == "retron/semantic_trace.csv"


def test_retron_sponge_protocol_rejects_dual_reporter_only_plot_selection(tmp_path: Path) -> None:
    cfg = write_config(
        tmp_path / "config.yaml",
        base_reader_config(
            experiment_id="retron_plot_guard",
            protocol_id="plate_reader/retron_sponge_screen",
            protocol_analysis={
                "semantic_metrics": {
                    "relevant_stress_map": {"sulAp": "100 nM ciprofloxacin"},
                    "sensor_target_map": {"sulAp": ["LexA"]},
                }
            },
            protocol_outputs={"plots": {"include": ["ratio_heatmap"]}},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["explain", str(cfg)])

    assert result.exit_code != 0
    assert "ratio_heatmap" in result.output
    assert "protocol.outputs.plots.include contains unknown deliverable" in result.output


def test_dual_reporter_ranking_references_declared_metrics() -> None:
    descriptor = builtin_protocol_catalog().resolve("plate_reader/dual_reporter_screen")
    ranking = descriptor.ranking
    metric_ids = {item.id for item in descriptor.metrics}

    assert ranking is not None
    assert ranking.primary_metric in metric_ids
    assert set(ranking.penalties).issubset(metric_ids)
    assert set(ranking.supporting_metrics).issubset(metric_ids)


def test_cytometry_protocol_rejects_unsupported_plot_selection(tmp_path: Path) -> None:
    cfg = write_config(
        tmp_path / "config.yaml",
        base_reader_config(
            experiment_id="cyto",
            protocol_id="cytometry/flow_panel",
            protocol_inputs={
                "ingest": {"auto_roots": ["./inputs"]},
                "metadata": {"require_columns": ["design_id", "treatment"]},
            },
            protocol_outputs={"plots": {"include": ["cytometry_qc"]}},
            resources={"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}},
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["explain", str(cfg)])

    assert result.exit_code != 0
    assert "cytometry/flow_panel does not currently compile plot outputs" in result.output
    assert "cytometry_qc" in result.output


def test_plugins_command_can_emit_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["plugins", "--protocol", "plate_reader/dual_reporter_screen", "--category", "transform", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["selection"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["selection"]["category"] == "transform"
    assert payload["summary"]["plugins"] >= 1
    assert set(payload["summary"]["by_category"]) == {"transform"}
    assert all(item["category"] == "transform" for item in payload["plugins"])
    assert "protocol" not in payload
    assert "count" not in payload


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
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["catalog"]["path"].endswith("outputs/manifests/records.json")
    assert payload["selection"]["include_history"] is True
    assert payload["summary"]["records"] == 1
    assert payload["summary"]["history"]["included"] is True
    assert payload["summary"]["history"]["revisions"] == 2
    assert payload["summary"]["by_kind"] == {"dataframe_artifact": 1}
    assert payload["summary"]["by_producer"] == {"pipeline:ingest": 1}
    assert payload["records"][0]["record_id"] == "ingest/df"
    assert payload["records"][0]["revision_count"] == 2
    assert "all" not in payload
    assert "count" not in payload


def test_records_command_can_emit_json_without_history_summary(tmp_path: Path) -> None:
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

    runner = CliRunner()
    result = runner.invoke(cli.app, ["records", str(cfg_path), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["id"] == "exp"
    assert payload["selection"]["include_history"] is False
    assert payload["summary"]["records"] == 1
    assert payload["summary"]["history"]["included"] is False
    assert payload["summary"]["history"]["revisions"] is None
    assert payload["records"][0]["record_id"] == "ingest/df"
    assert "revision_count" not in payload["records"][0]
