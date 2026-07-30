from __future__ import annotations

import importlib
from typing import cast

from reader_workbench.workbench.assets import AssetCatalog, AssetDescriptor

_CATEGORY_MODULES = {
    "ingest": "reader_workbench.plugins.manifests.ingest",
    "transform": "reader_workbench.plugins.manifests.transform",
    "plot": "reader_workbench.plugins.manifests.plot",
    "export": "reader_workbench.plugins.manifests.export",
    "validator": "reader_workbench.plugins.manifests.validator",
}


def _category_descriptors(category: str) -> tuple[AssetDescriptor, ...]:
    module = importlib.import_module(_CATEGORY_MODULES[category])
    return cast(tuple[AssetDescriptor, ...], module.BUILTIN_PLUGIN_DESCRIPTORS)


def builtin_plugin_catalog(*, categories: set[str] | None = None) -> AssetCatalog:
    selected = (
        tuple(_CATEGORY_MODULES)
        if categories is None
        else tuple(category for category in _CATEGORY_MODULES if category in categories)
    )
    descriptors = [descriptor for category in selected for descriptor in _category_descriptors(category)]
    return AssetCatalog(descriptors)


def builtin_plugin_descriptors(*, categories: set[str] | None = None) -> tuple[AssetDescriptor, ...]:
    return builtin_plugin_catalog(categories=categories).all()
