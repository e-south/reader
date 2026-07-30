from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from reader_workbench.tests.support import REPO_ROOT, cli_success_data
from reader_workbench.workbench.cli.shared import app


def test_maintainer_checks_are_discoverable_from_one_nested_surface() -> None:
    result = CliRunner().invoke(app, ["maintain", "--help"])

    assert result.exit_code == 0, result.output
    assert "docs" in result.output
    assert "skills" in result.output


def test_docs_maintenance_check_returns_typed_payload() -> None:
    result = CliRunner().invoke(
        app,
        ["maintain", "docs", "--repo-root", str(REPO_ROOT), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = cli_success_data(result.output)
    assert payload["schema"] == "reader.maintenance/v1"
    assert payload["check"] == "docs"
    assert payload["status"] == "ok"
    assert payload["checked"] > 0


def test_failed_maintenance_check_uses_process_json_error_contract(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='reader-test'\n", encoding="utf-8")
    (tmp_path / "src" / "reader_workbench").mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reader_workbench",
            "maintain",
            "skills",
            "--repo-root",
            str(tmp_path),
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["schema"] == "reader.cli/v1"
    assert payload["ok"] is False
    assert payload["data"] is None
    assert payload["error"]["code"] == "maintenance_check_failed"
    assert payload["error"]["field"] == "repo_root"
