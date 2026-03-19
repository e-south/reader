from __future__ import annotations

import importlib

__all__ = ["write_experiment_notebook"]


def __getattr__(name: str):
    if name == "write_experiment_notebook":
        module = importlib.import_module("reader.workbench.notebooks.scaffold")
        return getattr(module, name)
    raise AttributeError(name)
