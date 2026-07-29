from __future__ import annotations

from reader.workbench.templates import builtin_notebook_template_catalog

from .types import AssetCatalog


def _build_static_asset_catalog() -> AssetCatalog:
    descriptors = [template.as_asset() for template in builtin_notebook_template_catalog().all()]
    return AssetCatalog(descriptors)


_STATIC_ASSET_CATALOG = _build_static_asset_catalog()


def static_asset_catalog() -> AssetCatalog:
    return _STATIC_ASSET_CATALOG
