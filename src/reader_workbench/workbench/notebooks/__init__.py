from __future__ import annotations

import importlib

_EXPORTS = {
    "scaffold": {"CANONICAL_NOTEBOOK_ID", "write_experiment_notebook"},
}

__all__ = tuple(sorted({name for names in _EXPORTS.values() for name in names}))


def __getattr__(name: str):
    for module_name, names in _EXPORTS.items():
        if name in names:
            module = importlib.import_module(f"reader_workbench.workbench.notebooks.{module_name}")
            return getattr(module, name)
    raise AttributeError(name)
