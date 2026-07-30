from __future__ import annotations

from functools import cache

from reader_workbench.contracts import builtin_contract_catalog
from reader_workbench.plotting.mpl import ensure_mpl_cache_dir
from reader_workbench.plugins.catalog import builtin_plugin_descriptors
from reader_workbench.protocols.builtins import builtin_protocol_catalog
from reader_workbench.workbench.registry import load_plugin_catalog

from .model import ReaderRuntime


@cache
def builtin_runtime() -> ReaderRuntime:
    ensure_mpl_cache_dir()
    contracts = builtin_contract_catalog()
    protocols = builtin_protocol_catalog()
    plugins = load_plugin_catalog(
        contracts=contracts,
        builtin_descriptors=builtin_plugin_descriptors(),
    )
    return ReaderRuntime(contracts=contracts, protocols=protocols, plugins=plugins)
