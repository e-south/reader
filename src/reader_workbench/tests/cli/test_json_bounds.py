from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from reader_workbench.contracts import builtin_contract_catalog
from reader_workbench.tests.support import base_reader_config, write_config
from reader_workbench.workbench import cli
from reader_workbench.workbench.records import RecordStore

REPO_ROOT = Path(__file__).resolve().parents[4]
CONSOLE_SCRIPT = REPO_ROOT / ".venv" / "bin" / "reader"


def _envelope(output: str) -> dict[str, object]:
    payload = json.loads(output)
    assert payload["schema"] == "reader.cli/v1"
    return payload


def _write_experiment(root: Path, name: str) -> Path:
    experiment = root / name
    experiment.mkdir(parents=True)
    return write_config(
        experiment / "config.yaml",
        base_reader_config(experiment_id=name),
    )


@pytest.mark.parametrize(
    ("section", "expected_keys"),
    [
        ("identity", {"experiment"}),
        ("authoring", {"experiment", "authoring"}),
        ("semantics", {"experiment", "semantics"}),
        ("plan", {"experiment", "plan"}),
        ("compiled", {"experiment", "compiled"}),
        ("inputs", {"experiment", "inputs"}),
        ("generated", {"experiment", "generated"}),
        ("readiness", {"experiment", "readiness"}),
    ],
)
def test_inspect_json_supports_stable_semantic_sections(
    tmp_path: Path,
    section: str,
    expected_keys: set[str],
) -> None:
    config = _write_experiment(tmp_path, "bounded_inspect")

    result = CliRunner().invoke(
        cli.app,
        ["inspect", str(config), "--section", section, "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    envelope = _envelope(result.output)
    assert envelope["ok"] is True
    assert envelope["meta"] == {
        "projection": f"section:{section}",
        "truncated": False,
        "continuation": None,
    }
    assert set(envelope["data"]) == expected_keys
    assert envelope["data"]["experiment"]["id"] == "bounded_inspect"


@pytest.mark.parametrize(
    ("section", "expected_keys"),
    [
        ("identity", {"protocol", "domain", "family", "summary", "tags"}),
        ("authoring", {"protocol", "domain", "family", "summary", "tags", "authoring"}),
        ("semantics", {"protocol", "domain", "family", "summary", "tags", "semantics"}),
        ("defaults", {"protocol", "domain", "family", "summary", "tags", "defaults"}),
        ("compiled", {"protocol", "domain", "family", "summary", "tags", "compiled"}),
    ],
)
def test_named_protocol_json_supports_stable_semantic_sections(
    section: str,
    expected_keys: set[str],
) -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "protocols",
            "plate_reader/dual_reporter_screen",
            "--section",
            section,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    envelope = _envelope(result.output)
    assert envelope["meta"]["projection"] == f"section:{section}"
    assert set(envelope["data"]) == expected_keys
    assert envelope["data"]["protocol"] == "plate_reader/dual_reporter_screen"


def test_ls_json_pages_experiments_in_stable_key_order(tmp_path: Path) -> None:
    for name in ("exp_c", "exp_a", "exp_e", "exp_b", "exp_d"):
        _write_experiment(tmp_path, name)
    runner = CliRunner()

    first = runner.invoke(
        cli.app,
        ["ls", "--root", str(tmp_path), "--limit", "2", "--format", "json"],
    )

    assert first.exit_code == 0, first.output
    first_envelope = _envelope(first.output)
    assert first_envelope["meta"]["truncated"] is True
    continuation = first_envelope["meta"]["continuation"]
    assert isinstance(continuation, str) and continuation
    assert first_envelope["data"]["summary"]["experiments"] == 5
    assert [item["name"] for item in first_envelope["data"]["experiments"]] == ["exp_a", "exp_b"]

    second = runner.invoke(
        cli.app,
        [
            "ls",
            "--root",
            str(tmp_path),
            "--limit",
            "2",
            "--continuation",
            continuation,
            "--format",
            "json",
        ],
    )

    assert second.exit_code == 0, second.output
    second_envelope = _envelope(second.output)
    assert [item["name"] for item in second_envelope["data"]["experiments"]] == ["exp_c", "exp_d"]
    assert second_envelope["meta"]["truncated"] is True
    assert second_envelope["meta"]["continuation"] != continuation


def test_ls_json_applies_default_collection_bound(tmp_path: Path) -> None:
    for index in range(27):
        _write_experiment(tmp_path, f"exp_{index:02d}")

    result = CliRunner().invoke(
        cli.app,
        ["ls", "--root", str(tmp_path), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    envelope = _envelope(result.output)
    assert envelope["data"]["summary"]["experiments"] == 27
    assert len(envelope["data"]["experiments"]) == 25
    assert envelope["meta"]["truncated"] is True
    assert envelope["meta"]["continuation"]


def test_plugins_json_pages_without_changing_total_summary() -> None:
    runner = CliRunner()
    first = runner.invoke(cli.app, ["plugins", "--limit", "2", "--format", "json"])

    assert first.exit_code == 0, first.output
    first_envelope = _envelope(first.output)
    plugins = first_envelope["data"]["plugins"]
    assert len(plugins) == 2
    assert [item["plugin"] for item in plugins] == sorted(item["plugin"] for item in plugins)
    assert first_envelope["data"]["summary"]["plugins"] > len(plugins)
    assert first_envelope["meta"]["truncated"] is True


def test_records_json_pages_latest_records_by_record_id(tmp_path: Path) -> None:
    config = _write_experiment(tmp_path, "bounded_records")
    store = RecordStore(tmp_path / "bounded_records" / "outputs", contracts=builtin_contract_catalog())
    frame = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["OD600"],
            "value": [1.0],
        }
    )
    for record_id in ("step/z", "step/a", "step/m"):
        store.persist_dataframe(
            producer_id=record_id.replace("/", "_"),
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id=record_id,
            df=frame,
            contract_id="tidy.v1",
            inputs=[],
            config_digest="sha256:test",
        )
    runner = CliRunner()

    first = runner.invoke(
        cli.app,
        ["records", str(config), "--limit", "2", "--format", "json"],
    )

    assert first.exit_code == 0, first.output
    first_envelope = _envelope(first.output)
    assert [item["record_id"] for item in first_envelope["data"]["records"]] == ["step/a", "step/m"]
    assert first_envelope["data"]["summary"]["records"] == 3
    assert first_envelope["meta"]["truncated"] is True
    continuation = first_envelope["meta"]["continuation"]

    second = runner.invoke(
        cli.app,
        [
            "records",
            str(config),
            "--limit",
            "2",
            "--continuation",
            continuation,
            "--format",
            "json",
        ],
    )

    assert second.exit_code == 0, second.output
    second_envelope = _envelope(second.output)
    assert [item["record_id"] for item in second_envelope["data"]["records"]] == ["step/z"]
    assert second_envelope["meta"] == {
        "projection": "full",
        "truncated": False,
        "continuation": None,
    }


@pytest.mark.parametrize(
    ("args", "field"),
    [
        (("inspect", "CONFIG", "--section", "raw.path"), "section"),
        (("protocols", "plate_reader/dual_reporter_screen", "--section", "raw.path"), "section"),
        (("plugins", "--limit", "0"), "limit"),
        (("plugins", "--continuation", "not-a-reader-token"), "continuation"),
    ],
)
def test_invalid_json_bounds_are_structured(
    tmp_path: Path,
    args: tuple[str, ...],
    field: str,
) -> None:
    config = _write_experiment(tmp_path, "invalid_bounds")
    rendered = tuple(str(config) if item == "CONFIG" else item for item in args)

    result = subprocess.run(
        [str(CONSOLE_SCRIPT), *rendered, "--format", "json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stderr == ""
    envelope = _envelope(result.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "invalid_parameter"
    assert envelope["error"]["field"] == field


def test_continuation_is_bound_to_plugin_selection() -> None:
    runner = CliRunner()
    first = runner.invoke(cli.app, ["plugins", "--limit", "1", "--format", "json"])
    continuation = _envelope(first.output)["meta"]["continuation"]

    result = subprocess.run(
        [
            str(CONSOLE_SCRIPT),
            "plugins",
            "--category",
            "transform",
            "--limit",
            "1",
            "--continuation",
            continuation,
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stderr == ""
    envelope = _envelope(result.stdout)
    assert envelope["error"]["field"] == "continuation"
