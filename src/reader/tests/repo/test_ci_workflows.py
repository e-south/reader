from __future__ import annotations

import re
from typing import Any

import yaml

from reader.tests.support import REPO_ROOT

_PINNED_ACTION = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}")
_WORKFLOWS = ("checks.yaml", "release.yaml")


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


def test_checks_is_the_only_continuous_workflow() -> None:
    triggers = _triggers(_workflow("checks.yaml"))

    assert triggers["push"] == {"branches": ["main"]}
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers
    assert "schedule" not in triggers


def test_release_is_published_release_only() -> None:
    workflow = _workflow("release.yaml")
    triggers = _triggers(workflow)

    assert triggers == {"release": {"types": ["published"]}}
    verify_step = next(
        step for step in workflow["jobs"]["build"]["steps"] if step.get("name") == "Verify release version"
    )
    assert 'test "${GITHUB_REF_NAME}" = "v${project_version}"' in verify_step["run"]


def test_hidden_test_results_are_uploaded_or_fail_explicitly() -> None:
    steps = _workflow("checks.yaml")["jobs"]["tests"]["steps"]
    upload = next(step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@"))

    assert upload["with"]["path"] == ".artifacts/"
    assert upload["with"]["include-hidden-files"] is True
    assert upload["with"]["if-no-files-found"] == "error"


def test_parallel_jobs_use_one_uv_cache_writer() -> None:
    jobs = _workflow("checks.yaml")["jobs"]
    setup_steps = {
        job_name: next(step for step in jobs[job_name]["steps"] if "astral-sh/setup-uv@" in str(step.get("uses", "")))
        for job_name in ("package", "tests")
    }

    assert setup_steps["package"]["with"]["save-cache"] is False
    assert "save-cache" not in setup_steps["tests"]["with"]


def test_dependency_audit_covers_runtime_and_notebook_surfaces() -> None:
    steps = _workflow("checks.yaml")["jobs"]["package"]["steps"]
    audit = next(step for step in steps if step.get("name") == "Audit operational dependencies")

    assert "--no-dev --group notebooks --no-emit-project" in audit["run"]
    assert "--no-deps" in audit["run"]
    assert "--disable-pip" in audit["run"]


def test_release_oidc_is_limited_to_publish_job() -> None:
    jobs = _workflow("release.yaml")["jobs"]

    assert jobs["build"].get("permissions") is None
    assert jobs["publish"]["permissions"] == {"id-token": "write"}
    assert jobs["publish"]["environment"] == "pypi"


def test_workflow_actions_are_pinned_to_immutable_commits() -> None:
    for workflow_name in _WORKFLOWS:
        for job in _workflow(workflow_name)["jobs"].values():
            for step in job["steps"]:
                uses = step.get("uses")
                if uses is not None:
                    assert _PINNED_ACTION.fullmatch(uses), f"{workflow_name}: unpinned action {uses!r}"
