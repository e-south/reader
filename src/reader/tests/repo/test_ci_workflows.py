from __future__ import annotations

import re
from typing import Any

import yaml

from reader.tests.support import REPO_ROOT

_PINNED_ACTION = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}")


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
        upload = next(step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@"))

        assert upload["with"]["path"] == ".artifacts/"
        assert upload["with"]["include-hidden-files"] is True
        assert upload["with"]["if-no-files-found"] == "error"


def test_workflow_actions_are_pinned_to_immutable_commits() -> None:
    for workflow_name in ("ci.yaml", "integration.yaml"):
        jobs = _workflow(workflow_name)["jobs"]
        for job in jobs.values():
            for step in job["steps"]:
                uses = step.get("uses")
                if uses is not None:
                    assert _PINNED_ACTION.fullmatch(uses), f"{workflow_name}: unpinned action {uses!r}"
