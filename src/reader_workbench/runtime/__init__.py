from __future__ import annotations

from importlib import import_module

from .model import ReaderRuntime


def builtin_runtime() -> ReaderRuntime:
    return import_module("reader_workbench.runtime.builtin").builtin_runtime()


__all__ = ["ReaderRuntime", "builtin_runtime"]
