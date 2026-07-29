from __future__ import annotations

from functools import cache

from reader.contracts import builtin_contract_catalog
from reader.plotting.mpl import ensure_mpl_cache_dir
from reader.protocols.builtins import builtin_protocol_catalog
from reader.workbench.registry import load_plugin_catalog

from .model import ReaderRuntime


@cache
def builtin_runtime() -> ReaderRuntime:
    ensure_mpl_cache_dir()
    contracts = builtin_contract_catalog()
    protocols = builtin_protocol_catalog()
    plugins = load_plugin_catalog(contracts=contracts)
    return ReaderRuntime(contracts=contracts, protocols=protocols, plugins=plugins)
