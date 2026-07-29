from __future__ import annotations

from typing import Any

import yaml

from reader.tests.support import REPO_ROOT


def _workflow(name: str) -> dict[str, Any]:
    path = REPO_ROOT / ".github" / "workflows" / name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML follows YAML 1.1 and parses the unquoted GitHub key `on` as true.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return triggers


def test_main_push_runs_default_ci_without_duplicate_integration_job() -> None:
    ci_triggers = _triggers(_workflow("ci.yaml"))
    integration_triggers = _triggers(_workflow("integration.yaml"))

    assert ci_triggers["push"] == {"branches": ["main"]}
    assert "pull_request" in ci_triggers
    assert "push" not in integration_triggers
    assert "schedule" in integration_triggers
    assert "workflow_dispatch" in integration_triggers


def test_hidden_test_artifact_bundles_are_uploaded_or_fail_explicitly() -> None:
    for workflow_name, job_name in (("ci.yaml", "default-tests"), ("integration.yaml", "integration")):
        steps = _workflow(workflow_name)["jobs"][job_name]["steps"]
        upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")

        assert upload["with"]["path"] == ".artifacts/"
        assert upload["with"]["include-hidden-files"] is True
        assert upload["with"]["if-no-files-found"] == "error"
