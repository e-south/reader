from __future__ import annotations

from reader.workbench.notebooks.runtime_patch import _build_safe_service_worker_script


def test_safe_service_worker_script_guards_worker_activation() -> None:
    script = _build_safe_service_worker_script("demo/notebook.py")

    assert "registration?.active" in script
    assert "registration?.waiting" in script
    assert "registration?.installing" in script
    assert "registration.active.postMessage" not in script
    assert "worker.state === 'activated'" in script
    assert "statechange" in script
