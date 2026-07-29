from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import reader.workbench.audit.experiments as audit_module
from reader.runtime import builtin_runtime
from reader.tests.support import REPO_ROOT, base_reader_config, cli_error_data, cli_success_data, write_config
from reader.workbench.cli.shared import app


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_audit_rejects_external_discovered_input_symlink_before_staging(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiments" / "2027" / "exp_symlink_escape"
    inputs = experiment_dir / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "metadata.csv").write_text(
        "position,design_id,treatment\nA1,d1,control\n",
        encoding="utf-8",
    )
    outside_raw = tmp_path / "outside_raw.xlsx"
    outside_raw.write_text("must not be copied", encoding="utf-8")
    _symlink_or_skip(inputs / "raw.xlsx", outside_raw)
    config = write_config(
        experiment_dir,
        base_reader_config(
            experiment_id="exp_symlink_escape",
            lifecycle="active",
            protocol_id="plate_reader/dual_reporter_screen",
            protocol_analysis={"include_fold_change": False},
            protocol_outputs={"plots": {"profile": "none"}},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.csv"}},
        ),
    )

    result = audit_module.audit_config(config, include_non_active=False, runtime=builtin_runtime())

    assert result.status == "failed"
    assert result.phase == "validate"
    assert "must stay under the experiment root after resolving symlinks" in str(result.detail)


def test_stage_experiment_preserves_external_links_and_retargets_internal_absolute_links(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "inputs").mkdir(parents=True)
    (source / "assets").mkdir()
    internal_target = source / "assets" / "raw.xlsx"
    internal_target.write_text("internal", encoding="utf-8")
    outside_target = tmp_path / "outside.xlsx"
    outside_target.write_text("external", encoding="utf-8")
    _symlink_or_skip(source / "inputs" / "internal.xlsx", internal_target.resolve())
    _symlink_or_skip(source / "inputs" / "external.xlsx", outside_target.resolve())
    (source / "config.yaml").write_text(
        "schema: reader/v8\nexperiment:\n  id: staged\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    target = tmp_path / "staged"

    audit_module.stage_experiment(source, target)

    staged_internal = target / "inputs" / "internal.xlsx"
    staged_external = target / "inputs" / "external.xlsx"
    assert staged_internal.is_symlink()
    assert staged_internal.resolve() == target / "assets" / "raw.xlsx"
    assert staged_external.is_symlink()
    assert staged_external.resolve() == outside_target


def test_audit_verification_uses_composed_runtime_record_store(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    manifests = outputs / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "records.json").write_text("{}", encoding="utf-8")
    store = SimpleNamespace(iter_latest_records=lambda: [SimpleNamespace(record_id="ingest/df")])
    calls: list[dict[str, object]] = []

    class RuntimeProbe:
        def record_store(self, path: Path, **kwargs):
            calls.append({"path": path, **kwargs})
            return store

    layout = SimpleNamespace(outputs_dir=outputs, plots_subdir="plots", exports_subdir="exports")
    decl = SimpleNamespace(
        experiment_semantics=SimpleNamespace(layout=layout),
        experiment=SimpleNamespace(root=tmp_path),
        config_digest="sha256:audit-test",
    )
    monkeypatch.setattr(
        audit_module,
        "resolve_workbench",
        lambda _decl: SimpleNamespace(plots=(), exports=(), notebooks=(), plugin_steps=lambda: ()),
    )
    monkeypatch.setattr(audit_module, "verify_record_store", lambda *_args, **_kwargs: {"status": "ok"})

    result = audit_module.verify_outputs(decl, RuntimeProbe())

    assert result == (0, 0, None)
    assert calls == [
        {
            "path": outputs,
            "plots_subdir": "plots",
            "exports_subdir": "exports",
            "experiment_root": tmp_path,
            "create": False,
        }
    ]


def test_audit_local_experiments_auto_discovers_numeric_year_dirs(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiment_dir = experiments_root / "2027" / "exp_auto"
    experiment_dir.mkdir(parents=True)
    (experiments_root / "template").mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text(
        "schema: reader/v8\nexperiment:\n  id: exp_auto\n  lifecycle: draft\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["audit", "experiments", "--root", str(experiments_root), "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = cli_success_data(result.output)
    assert payload["years"] == ["2027"]
    assert payload["summary"] == {"experiments": 1, "passed": 0, "failed": 0, "skipped": 1}
    assert payload["results"][0]["config"].endswith("2027/exp_auto/config.yaml")


def test_audit_local_experiments_include_non_active_flag(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiment_dir = experiments_root / "2027" / "exp_auto"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text(
        "schema: reader/v8\nexperiment:\n  id: exp_auto\n  lifecycle: draft\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reader",
            "audit",
            "experiments",
            "--root",
            str(experiments_root),
            "--format",
            "json",
            "--include-non-active",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["schema"] == "reader.cli/v1"
    assert payload["ok"] is False
    assert payload["data"] is None
    assert payload["error"]["code"] == "experiment_audit_failed"
    assert payload["error"]["field"] == "experiments"
    assert "exp_auto" in payload["error"]["reason"]


def test_audit_local_experiments_does_not_mutate_source_outputs(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiment_dir = experiments_root / "2027" / "exp_active"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text(
        "schema: reader/v8\nexperiment:\n  id: exp_active\n  lifecycle: active\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    outputs_dir = experiment_dir / "outputs"
    outputs_dir.mkdir()
    sentinel = outputs_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = CliRunner().invoke(app, ["audit", "experiments", "--root", str(experiments_root), "--format", "json"])

    assert result.exit_code == 1, result.output
    error = cli_error_data(result.output)
    assert error["code"] == "experiment_audit_failed"
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (outputs_dir / "manifests").exists()
