from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
TOOL_PATH = REPO_ROOT / "tools" / "audit_local_experiments.py"


def test_audit_local_experiments_auto_discovers_numeric_year_dirs(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiment_dir = experiments_root / "2027" / "exp_auto"
    experiment_dir.mkdir(parents=True)
    (experiments_root / "template").mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text(
        "schema: reader/v7\nexperiment:\n  id: exp_auto\n  lifecycle: draft\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--root", str(experiments_root), "--format", "json"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["years"] == ["2027"]
    assert payload["summary"] == {"experiments": 1, "passed": 0, "failed": 0, "skipped": 1}
    assert payload["results"][0]["config"].endswith("2027/exp_auto/config.yaml")


def test_audit_local_experiments_include_non_active_flag(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiment_dir = experiments_root / "2027" / "exp_auto"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text(
        "schema: reader/v7\nexperiment:\n  id: exp_auto\n  lifecycle: draft\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--root", str(experiments_root), "--format", "json", "--include-non-active"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"] == {"experiments": 1, "passed": 0, "failed": 1, "skipped": 0}
    assert payload["results"][0]["status"] == "failed"


def test_audit_local_experiments_does_not_mutate_source_outputs(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiment_dir = experiments_root / "2027" / "exp_active"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.yaml").write_text(
        "schema: reader/v7\nexperiment:\n  id: exp_active\n  lifecycle: active\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    outputs_dir = experiment_dir / "outputs"
    outputs_dir.mkdir()
    sentinel = outputs_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--root", str(experiments_root), "--format", "json"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (outputs_dir / "manifests").exists()
