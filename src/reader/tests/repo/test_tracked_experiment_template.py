from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from reader.workbench.cli import app

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_tracked_experiment_template_passes_canonical_validation() -> None:
    result = CliRunner().invoke(
        app,
        ["validate", str(REPO_ROOT / "experiments" / "template"), "--no-files", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["data"]["summary"]["status"] == "ok"
