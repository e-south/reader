from __future__ import annotations

import importlib
from typing import cast

from .types import AssetCatalog, AssetDescriptor

_CATEGORY_MODULES = {
    "ingest": "reader.workbench.assets.plugin_manifests.ingest",
    "transform": "reader.workbench.assets.plugin_manifests.transform",
    "plot": "reader.workbench.assets.plugin_manifests.plot",
    "export": "reader.workbench.assets.plugin_manifests.export",
    "validator": "reader.workbench.assets.plugin_manifests.validator",
}


def _category_descriptors(category: str) -> tuple[AssetDescriptor, ...]:
    module = importlib.import_module(_CATEGORY_MODULES[category])
    return cast(tuple[AssetDescriptor, ...], module.BUILTIN_PLUGIN_DESCRIPTORS)


def builtin_plugin_asset_catalog(*, categories: set[str] | None = None) -> AssetCatalog:
    selected = (
        tuple(_CATEGORY_MODULES)
        if categories is None
        else tuple(category for category in _CATEGORY_MODULES if category in categories)
    )
    descriptors = [descriptor for category in selected for descriptor in _category_descriptors(category)]
    return AssetCatalog(descriptors)


def builtin_plugin_descriptors(*, categories: set[str] | None = None) -> tuple[AssetDescriptor, ...]:
    return builtin_plugin_asset_catalog(categories=categories).all()
