from __future__ import annotations

from functools import cache

from reader.contracts import builtin_contract_catalog
from reader.workbench.assets import build_workbench_asset_catalog
from reader.workbench.registry import load_plugin_catalog

from .model import ReaderRuntime


@cache
def builtin_runtime() -> ReaderRuntime:
    contracts = builtin_contract_catalog()
    plugins = load_plugin_catalog(contracts=contracts)
    assets = build_workbench_asset_catalog(plugin_registry=plugins)
    return ReaderRuntime(contracts=contracts, plugins=plugins, assets=assets)
