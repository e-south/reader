from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reader.tests.support import base_reader_config, write_config

REPO_ROOT = Path(__file__).resolve().parents[4]
CONSOLE_SCRIPT = REPO_ROOT / ".venv" / "bin" / "reader"


def _run_process(entrypoint: str, *args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "reader"] if entrypoint == "module" else [str(CONSOLE_SCRIPT)]
    return subprocess.run(
        [*command, *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_envelope(result: subprocess.CompletedProcess[str], *, ok: bool, command: str) -> dict[str, object]:
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload.keys() == {"schema", "ok", "command", "data", "error", "meta"}
    assert payload["schema"] == "reader.cli/v1"
    assert payload["ok"] is ok
    assert payload["command"] == command
    assert payload["meta"] == {"projection": "full", "truncated": False, "continuation": None}
    return payload


@pytest.mark.parametrize("entrypoint", ["console", "module"])
def test_process_entrypoints_emit_versioned_success_envelope(entrypoint: str, tmp_path: Path) -> None:
    result = _run_process(entrypoint, "ls", "--root", str(tmp_path), "--format", "json")

    assert result.returncode == 0
    payload = _assert_envelope(result, ok=True, command="ls")
    assert payload["error"] is None
    assert payload["data"]["experiments"] == []


@pytest.mark.parametrize(
    ("args", "code", "field"),
    [
        (("ls", "--root", "/definitely/missing", "--format", "json"), "invalid_parameter", "root"),
        (("protocols", "missing/protocol", "--format", "json"), "invalid_parameter", "name"),
        (("ls", "--unknown", "value", "--format", "json"), "usage_error", "arguments"),
    ],
)
def test_json_failures_are_one_document_on_stdout(
    args: tuple[str, ...],
    code: str,
    field: str,
) -> None:
    result = _run_process("console", *args)

    assert result.returncode != 0
    payload = _assert_envelope(result, ok=False, command=args[0])
    assert payload["data"] is None
    assert payload["error"].keys() == {"code", "field", "reason", "remediation", "retryable"}
    assert payload["error"]["code"] == code
    assert payload["error"]["field"] == field
    assert payload["error"]["reason"]
    assert payload["error"]["remediation"]
    assert payload["error"]["retryable"] is False


def test_json_reader_failure_is_structured_and_has_no_side_effects(tmp_path: Path) -> None:
    config = write_config(tmp_path, base_reader_config(experiment_id="json_failure"))

    result = _run_process(
        "console",
        "run",
        str(config),
        "--dry-run",
        "--log-level",
        "LOUD",
        "--format",
        "json",
    )

    assert result.returncode != 0
    payload = _assert_envelope(result, ok=False, command="run")
    assert payload["error"]["code"] == "reader_error"
    assert payload["error"]["field"] == "log_level"
    assert "LOUD" in payload["error"]["reason"]
    assert not (tmp_path / "outputs").exists()


def test_validation_failure_uses_error_envelope(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="json_validation",
            protocol_id="plate_reader/dual_reporter_screen",
            protocol_inputs={"fold_change": {"report_times": [14.0]}},
            resources={"sample_map": {"kind": "file", "path": "./inputs/missing.xlsx"}},
        ),
    )

    result = _run_process("console", "validate", str(config), "--format", "json")

    assert result.returncode != 0
    payload = _assert_envelope(result, ok=False, command="validate")
    assert payload["error"]["code"] == "validation_failed"
    assert payload["error"]["field"] == "experiment"


@pytest.mark.parametrize("surface", ["plot", "export"])
def test_surface_json_dry_run_is_enveloped_and_does_not_write(surface: str, tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="json_surface",
            protocol_id="plate_reader/dual_reporter_screen",
            protocol_inputs={"fold_change": {"report_times": [14.0]}},
            protocol_analysis={"crosstalk_pairs": {"enabled": True, "export": True}},
            protocol_outputs={
                "plots": {"profile": "none", "include": ["raw_kinetics"]},
                "exports": {"include": ["crosstalk_pairs_table"]},
            },
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        ),
    )

    result = _run_process("console", surface, str(config), "--dry-run", "--format", "json")

    assert result.returncode == 0
    payload = _assert_envelope(result, ok=True, command=surface)
    assert payload["data"]["dry_run"] is True
    assert payload["data"][f"{surface}s"][0]["id"]
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("surface", ["plot", "export"])
def test_surface_json_dry_run_rejects_invalid_log_level_without_writes(surface: str, tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="json_surface_bad_log",
            protocol_id="plate_reader/dual_reporter_screen",
            protocol_inputs={"fold_change": {"report_times": [14.0]}},
            protocol_analysis={"crosstalk_pairs": {"enabled": True, "export": True}},
            protocol_outputs={
                "plots": {"profile": "none", "include": ["raw_kinetics"]},
                "exports": {"include": ["crosstalk_pairs_table"]},
            },
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        ),
    )

    result = _run_process(
        "console",
        surface,
        str(config),
        "--dry-run",
        "--log-level",
        "LOUD",
        "--format",
        "json",
    )

    assert result.returncode != 0
    payload = _assert_envelope(result, ok=False, command=surface)
    assert payload["error"]["field"] == "log_level"
    assert not (tmp_path / "outputs").exists()
