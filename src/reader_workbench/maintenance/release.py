"""Fail closed unless a release commit passed the canonical main-push checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def require_successful_checks(payload: dict[str, Any], revision: str) -> dict[str, Any]:
    """Return the latest qualifying CI run; an older success cannot mask a failure."""
    runs = [
        run
        for run in payload["workflow_runs"]
        if run.get("head_sha") == revision
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
        and run.get("path") == ".github/workflows/checks.yaml"
        and (run.get("head_repository") or {}).get("full_name") == "e-south/reader"
    ]
    if not runs:
        raise ValueError("No canonical main-push Checks run for the release commit")
    latest = max(runs, key=lambda run: int(run["id"]))
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise ValueError("The latest main-push Checks run has not completed successfully")
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(require_successful_checks(json.loads(args.checks.read_text()), args.revision), indent=2))


if __name__ == "__main__":
    main()
