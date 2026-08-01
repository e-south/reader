from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

import reader_workbench.workbench.audit.experiments as audit_module
import reader_workbench.workbench.audit.staging as audit_staging
from reader_workbench.contracts import builtin_contract_catalog
from reader_workbench.runtime import builtin_runtime
from reader_workbench.tests.support import REPO_ROOT, base_reader_config, cli_error_data, cli_success_data, write_config
from reader_workbench.workbench.cli.shared import app
from reader_workbench.workbench.decl import load_workbench_decl
from reader_workbench.workbench.records import RecordStore


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def _vector_collection_audit_case(tmp_path: Path, *, record_ids: tuple[str, ...]):
    experiment_dir = tmp_path / "experiments" / "2027" / "vector_collection"
    experiment_dir.mkdir(parents=True)
    config = write_config(
        experiment_dir,
        base_reader_config(
            experiment_id="vector_collection",
            protocol_id="logic/four_state_vector_collection",
            protocol_outputs={
                "plots": {"profile": "none"},
                "exports": {"exclude": ["vector_table"]},
            },
        ),
    )
    runtime = builtin_runtime()
    decl = load_workbench_decl(config, protocols=runtime.protocols)
    manifests = experiment_dir / "outputs" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "records.json").write_text("{}", encoding="utf-8")
    store = SimpleNamespace(
        iter_latest_records=lambda: [SimpleNamespace(record_id=record_id) for record_id in record_ids]
    )
    runtime_probe = SimpleNamespace(
        plugins=runtime.plugins,
        record_store=lambda _path, **_kwargs: store,
    )
    return decl, runtime_probe


def _four_state_vector_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": ["design"],
            "reference_design_id": ["reference"],
            "intensity_log2_offset_delta": [0.0],
            "r_logic": [4.0],
            "v00": [0.0],
            "v10": [0.2],
            "v01": [0.7],
            "v11": [1.0],
            "y00_star": [-1.0],
            "y10_star": [0.0],
            "y01_star": [1.0],
            "y11_star": [2.0],
            "flat_logic": [False],
        }
    )


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

    audit_staging.stage_experiment(source, target)

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


def test_audit_verification_uses_compiled_protocol_record_outputs(tmp_path: Path, monkeypatch) -> None:
    decl, runtime = _vector_collection_audit_case(
        tmp_path,
        record_ids=("four_state_vector_collection/vectors",),
    )
    monkeypatch.setattr(audit_module, "verify_record_store", lambda *_args, **_kwargs: {"status": "ok"})

    result = audit_module.verify_outputs(decl, runtime)

    assert result == (0, 0, None)


def test_audit_verification_reports_missing_compiled_protocol_record(tmp_path: Path, monkeypatch) -> None:
    decl, runtime = _vector_collection_audit_case(tmp_path, record_ids=("ingest/df",))
    monkeypatch.setattr(audit_module, "verify_record_store", lambda *_args, **_kwargs: {"status": "ok"})

    result = audit_module.verify_outputs(decl, runtime)

    assert result == (
        0,
        0,
        "missing declared dataframe records: ['four_state_vector_collection/vectors']",
    )


def test_audit_runs_record_resource_protocol_in_canonical_staged_workspace(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    provider_dir = experiments_root / "2027" / "provider"
    consumer_dir = experiments_root / "2027" / "consumer"
    provider_dir.mkdir(parents=True)
    consumer_dir.mkdir(parents=True)
    provider_config = write_config(
        provider_dir,
        base_reader_config(experiment_id="provider"),
    )
    runtime = builtin_runtime()
    provider_decl = load_workbench_decl(provider_config, protocols=runtime.protocols)
    provider_store = RecordStore(
        provider_dir / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=provider_dir,
    )
    provider_record = provider_store.persist_dataframe(
        producer_id="four_state_vector",
        producer_plugin="transform/four_state_vector",
        out_name="vector",
        record_id="four_state_vector/vector",
        df=_four_state_vector_frame(),
        contract_id="logic.four_state_vector.v1",
        inputs=(),
        config_digest=provider_decl.config_digest,
    )
    consumer_config = write_config(
        consumer_dir,
        base_reader_config(
            experiment_id="consumer",
            protocol_id="logic/four_state_vector_collection",
            protocol_inputs={"record_resources": ["provider_vector"]},
            protocol_outputs={
                "plots": {"profile": "none"},
                "exports": {"exclude": ["vector_table"]},
            },
            resources={
                "provider_vector": {
                    "kind": "record",
                    "experiment": "provider",
                    "record": "four_state_vector/vector",
                }
            },
        ),
    )
    assert provider_record.path.is_file()
    source_tree_before = {
        path.relative_to(provider_dir): path.read_bytes() for path in provider_dir.rglob("*") if path.is_file()
    }
    target_config_before = consumer_config.read_bytes()

    result = audit_module.audit_config(consumer_config, include_non_active=False, runtime=runtime)

    assert result.status == "passed", result
    assert result.phase == "complete"
    assert {
        path.relative_to(provider_dir): path.read_bytes() for path in provider_dir.rglob("*") if path.is_file()
    } == source_tree_before
    assert consumer_config.read_bytes() == target_config_before
    assert not (consumer_dir / "outputs").exists()


def test_audit_stages_nonaggregate_config_from_explicit_noncanonical_root(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "custom-root" / "2027" / "experiment"
    experiment_dir.mkdir(parents=True)
    config = write_config(experiment_dir, base_reader_config(experiment_id="experiment"))

    result = audit_module.audit_config(config, include_non_active=False, runtime=builtin_runtime())

    assert result.status == "failed"
    assert result.phase == "verify"
    assert "catalog.empty" in str(result.detail)
    assert not (experiment_dir / "outputs").exists()


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


def test_audit_local_experiments_reports_malformed_config_and_continues(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    broken_dir = experiments_root / "2027" / "a_broken"
    draft_dir = experiments_root / "2027" / "z_draft"
    broken_dir.mkdir(parents=True)
    draft_dir.mkdir(parents=True)
    (broken_dir / "config.yaml").write_text(
        "schema: reader/v8\nexperiment:\n  id: a_broken\n",
        encoding="utf-8",
    )
    write_config(
        draft_dir,
        base_reader_config(experiment_id="z_draft", lifecycle="draft"),
    )

    payload = audit_module.audit_experiments(experiments_root)

    assert payload["summary"] == {"experiments": 2, "passed": 0, "failed": 1, "skipped": 1}
    assert [item["name"] for item in payload["results"]] == ["a_broken", "z_draft"]
    broken, draft = payload["results"]
    assert broken["lifecycle"] == "unknown"
    assert broken["status"] == "failed"
    assert broken["phase"] == "config"
    assert "protocol" in str(broken["detail"])
    assert draft["status"] == "skipped"
    assert not (broken_dir / "outputs").exists()
    assert not (draft_dir / "outputs").exists()

    args = ["audit", "experiments", "--root", str(experiments_root), "--format", "json"]
    first_cli_result = CliRunner().invoke(app, args)
    second_cli_result = CliRunner().invoke(app, args)

    assert first_cli_result.exit_code == 1
    assert second_cli_result.exit_code == 1
    assert first_cli_result.output == second_cli_result.output
    error = cli_error_data(first_cli_result.output)
    assert error["code"] == "experiment_audit_failed"
    assert "a_broken during config" in str(error["reason"])
    assert "configuration, validation, execution, or verification failure" in str(error["remediation"])


def test_audit_local_experiments_reports_invalid_utf8_and_continues(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    broken_dir = experiments_root / "2027" / "a_invalid_utf8"
    draft_dir = experiments_root / "2027" / "z_draft"
    broken_dir.mkdir(parents=True)
    draft_dir.mkdir(parents=True)
    (broken_dir / "config.yaml").write_bytes(b"\xff\xfe")
    write_config(draft_dir, base_reader_config(experiment_id="z_draft", lifecycle="draft"))

    payload = audit_module.audit_experiments(experiments_root)

    assert payload["summary"] == {"experiments": 2, "passed": 0, "failed": 1, "skipped": 1}
    broken, draft = payload["results"]
    assert broken["name"] == "a_invalid_utf8"
    assert broken["lifecycle"] == "unknown"
    assert broken["phase"] == "config"
    assert "UTF-8" in str(broken["detail"])
    assert draft["name"] == "z_draft"
    assert draft["status"] == "skipped"


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
            "reader_workbench",
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
