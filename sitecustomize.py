from __future__ import annotations

import importlib
import os
import sys


def _install_reader_marimo_runtime_patch() -> None:
    if os.environ.get("READER_MARIMO_RUNTIME_PATCH") != "1":
        return
    try:
        runtime_patch = importlib.import_module("reader.workbench.notebooks.runtime_patch")
        runtime_patch.install_runtime_patches()
    except Exception as exc:  # pragma: no cover - startup fallback
        print(
            f"[reader] Failed to install marimo runtime patch: {exc}",
            file=sys.stderr,
        )


_install_reader_marimo_runtime_patch()
