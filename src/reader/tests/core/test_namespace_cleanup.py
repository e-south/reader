from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "reader.core",
        "reader.io",
        "reader.lib",
        "reader.domains.plate_reader.support",
        "reader.domains.logic.semantics",
        "reader.workbench.model",
        "reader.workbench.semantics",
        "reader.workbench.notebooks.catalog",
        "reader.workbench.notebooks.templates",
        "reader.workbench.resources",
    ],
)
def test_legacy_namespace_packages_are_not_importable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
