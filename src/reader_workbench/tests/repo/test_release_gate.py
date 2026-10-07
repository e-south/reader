from __future__ import annotations

import pytest

from reader_workbench.maintenance.release import require_successful_checks

REVISION = "a" * 40


def _run(**changes):
    return {
        "id": 12,
        "head_sha": REVISION,
        "event": "push",
        "head_branch": "main",
        "path": ".github/workflows/checks.yaml",
        "head_repository": {"full_name": "e-south/reader"},
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        **changes,
    }


def test_qualifies_exact_main_push_and_retains_ci_identity():
    run = _run()
    assert require_successful_checks({"workflow_runs": [run]}, REVISION) == run


@pytest.mark.parametrize(
    "changes",
    [
        {"head_sha": "b" * 40},
        {"event": "pull_request"},
        {"head_branch": "preview"},
        {"path": ".github/workflows/other.yaml"},
        {"head_repository": {"full_name": "someone/reader"}},
        {"status": "in_progress", "conclusion": None},
        {"conclusion": "failure"},
        {"conclusion": "cancelled"},
    ],
)
def test_rejects_unqualified_release_commit(changes):
    with pytest.raises(ValueError):
        require_successful_checks({"workflow_runs": [_run(**changes)]}, REVISION)


def test_does_not_hide_latest_failure_behind_older_success():
    with pytest.raises(ValueError, match="latest"):
        require_successful_checks({"workflow_runs": [_run(id=11), _run(id=12, conclusion="failure")]}, REVISION)


def test_missing_ci_is_not_release_evidence():
    with pytest.raises(ValueError):
        require_successful_checks({"workflow_runs": []}, REVISION)
