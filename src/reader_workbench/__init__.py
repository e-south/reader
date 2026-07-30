"""Reader's narrow, task-oriented public entrypoint."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reader_workbench.api import Experiment, open_experiment

__all__ = ["Experiment", "__version__", "open_experiment"]


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from reader_workbench._version import package_version  # noqa: PLC0415

        return package_version()
    if name not in __all__:
        raise AttributeError(f"module 'reader_workbench' has no attribute {name!r}")
    api = import_module("reader_workbench.api")
    return getattr(api, name)
