from __future__ import annotations

import json
from pathlib import Path

import pytest

from reader.errors import ConfigError
from reader.workbench.notebooks import launch


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    notebook = repo_root / "experiments" / "2026" / "exp" / "outputs" / "notebooks" / "EDA.py"
    notebook.parent.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='reader-test'\n", encoding="utf-8")
    (repo_root / "experiments" / "2026" / "exp" / "config.yaml").write_text(
        "schema: reader/v7\nexperiment:\n  id: exp\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    notebook.write_text("import marimo\n", encoding="utf-8")
    return repo_root, notebook


def test_plan_marimo_launch_uses_repo_local_runtime_dirs(tmp_path: Path) -> None:
    repo_root, notebook = _make_repo(tmp_path)

    plan = launch.plan_marimo_launch(mode="run", target=notebook, headless=True, base_env={})

    assert plan.runtime_paths.root == repo_root / ".cache" / "marimo"
    assert plan.env["XDG_CONFIG_HOME"] == str(repo_root / ".cache" / "marimo" / "xdg-config")
    assert plan.env["XDG_STATE_HOME"] == str(repo_root / ".cache" / "marimo" / "xdg-state")
    assert plan.env["XDG_CACHE_HOME"] == str(repo_root / ".cache" / "marimo" / "xdg-cache")
    assert plan.env["MPLCONFIGDIR"] == str(repo_root / ".cache" / "marimo" / "matplotlib")
    assert plan.env["READER_MARIMO_RUNTIME_PATCH"] == "1"
    assert str(repo_root) in plan.env["PYTHONPATH"]
    assert "--host" in plan.cmd
    assert "--port" in plan.cmd
    assert "--headless" in plan.cmd
    assert "--no-token" in plan.cmd


def test_plan_marimo_launch_reuses_live_session_for_same_notebook(monkeypatch, tmp_path: Path) -> None:
    _, notebook = _make_repo(tmp_path)
    runtime_paths = launch._runtime_paths_for_target(notebook)
    monkeypatch.setattr(launch, "_target_signature", lambda target: (11, 22))
    monkeypatch.setattr(launch, "_runtime_fingerprint", lambda repo_root: "fp-current")
    record = launch.MarimoSessionRecord(
        pid=1234,
        port=2718,
        host="127.0.0.1",
        mode="run",
        notebook=str(notebook.resolve()),
        experiment_root=str((notebook.parents[2]).resolve()),
        repo_root=str(notebook.parents[4].resolve()),
        launched_at=1.0,
        notebook_mtime_ns=11,
        notebook_size_bytes=22,
        runtime_fingerprint="fp-current",
    )
    runtime_paths.registry_path.write_text(json.dumps([launch.asdict(record)]), encoding="utf-8")
    monkeypatch.setattr(launch, "_pid_is_live", lambda pid: True)
    monkeypatch.setattr(launch, "_port_is_open", lambda host, port, timeout=0.15: True)

    plan = launch.plan_marimo_launch(mode="run", target=notebook, headless=True, base_env={})

    assert plan.reused_session == record
    assert plan.cmd == ()
    assert plan.url == "http://127.0.0.1:2718"


def test_plan_marimo_launch_restarts_stale_same_notebook_session_on_runtime_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _, notebook = _make_repo(tmp_path)
    runtime_paths = launch._runtime_paths_for_target(notebook)
    record = launch.MarimoSessionRecord(
        pid=2222,
        port=2718,
        host="127.0.0.1",
        mode="run",
        notebook=str(notebook.resolve()),
        experiment_root=str((notebook.parents[2]).resolve()),
        repo_root=str(notebook.parents[4].resolve()),
        launched_at=1.0,
        notebook_mtime_ns=11,
        notebook_size_bytes=22,
        runtime_fingerprint="fp-stale",
    )
    runtime_paths.registry_path.write_text(json.dumps([launch.asdict(record)]), encoding="utf-8")
    monkeypatch.setattr(launch, "_pid_is_live", lambda pid: True)
    monkeypatch.setattr(launch, "_target_signature", lambda target: (11, 22))
    monkeypatch.setattr(launch, "_runtime_fingerprint", lambda repo_root: "fp-current")
    terminated: list[int] = []

    def fake_port_is_open(host: str, port: int, timeout: float = 0.15) -> bool:
        del host, timeout
        return port == 2718 and 2222 not in terminated

    monkeypatch.setattr(launch, "_port_is_open", fake_port_is_open)
    monkeypatch.setattr(
        launch,
        "_terminate_pid",
        lambda pid, grace_seconds=1.0: terminated.append(pid) or True,
    )

    plan = launch.plan_marimo_launch(mode="run", target=notebook, headless=True, base_env={})

    assert terminated == [2222]
    assert plan.reused_session is None
    assert len(plan.terminated_sessions) == 1
    assert plan.port == 2718


def test_plan_marimo_launch_prunes_same_experiment_sessions(monkeypatch, tmp_path: Path) -> None:
    _, notebook = _make_repo(tmp_path)
    old_notebook = notebook.with_name("EDA_old.py")
    old_notebook.write_text("import marimo\n", encoding="utf-8")
    runtime_paths = launch._runtime_paths_for_target(notebook)
    record = launch.MarimoSessionRecord(
        pid=2222,
        port=2718,
        host="127.0.0.1",
        mode="run",
        notebook=str(old_notebook.resolve()),
        experiment_root=str((notebook.parents[2]).resolve()),
        repo_root=str(notebook.parents[4].resolve()),
        launched_at=1.0,
    )
    runtime_paths.registry_path.write_text(json.dumps([launch.asdict(record)]), encoding="utf-8")
    monkeypatch.setattr(launch, "_pid_is_live", lambda pid: True)
    monkeypatch.setattr(launch, "_port_is_open", lambda host, port, timeout=0.15: False)
    terminated: list[int] = []
    monkeypatch.setattr(
        launch,
        "_terminate_pid",
        lambda pid, grace_seconds=1.0: terminated.append(pid) or True,
    )

    plan = launch.plan_marimo_launch(mode="run", target=notebook, headless=True, base_env={})

    assert terminated == [2222]
    assert len(plan.terminated_sessions) == 1
    assert plan.port == 2718


def test_plan_marimo_launch_rejects_busy_explicit_port(monkeypatch, tmp_path: Path) -> None:
    _, notebook = _make_repo(tmp_path)
    monkeypatch.setattr(launch, "_port_is_open", lambda host, port, timeout=0.15: port == 9999)

    with pytest.raises(ConfigError):
        launch.plan_marimo_launch(
            mode="run",
            target=notebook,
            headless=True,
            preferred_port=9999,
            base_env={},
        )
