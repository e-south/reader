from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "reader_workbench.core",
        "reader_workbench.io",
        "reader_workbench.lib",
        "reader_workbench.notebook_presentation",
        "reader_workbench.notebook_review",
        "reader_workbench.domains.plate_reader.support",
        "reader_workbench.domains.logic.semantics",
        "reader_workbench.workbench.model",
        "reader_workbench.workbench.semantics",
        "reader_workbench.workbench.notebooks.catalog",
        "reader_workbench.workbench.notebooks.templates",
        "reader_workbench.workbench.recipes",
        "reader_workbench.workbench.resources",
    ],
)
def test_legacy_namespace_packages_are_not_importable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
