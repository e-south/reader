from __future__ import annotations

from typing import Any

from reader.errors import ConfigError
from reader.workbench.templates import builtin_notebook_template_catalog

from .types import AssetCatalog


def _build_static_asset_catalog() -> AssetCatalog:
    descriptors = [template.as_asset() for template in builtin_notebook_template_catalog().all()]
    return AssetCatalog(descriptors)


_STATIC_ASSET_CATALOG = _build_static_asset_catalog()


def static_asset_catalog() -> AssetCatalog:
    return _STATIC_ASSET_CATALOG


def build_workbench_asset_catalog(*, plugin_registry: Any | None = None) -> AssetCatalog:
    if plugin_registry is None:
        raise ConfigError("build_workbench_asset_catalog() requires an explicit plugin registry.")
    plugin_assets = list(plugin_registry.catalog().all())
    return AssetCatalog(plugin_assets + list(static_asset_catalog().all()))
