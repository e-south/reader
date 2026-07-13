"""
--------------------------------------------------------------------------------
<reader project>
src/reader/contracts/builtins/__init__.py

Explicit built-in dataframe contract catalog.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import cache

from ..catalog import ContractCatalog
from .cytometry import CONTRACTS as CYTOMETRY_CONTRACTS
from .generic import CONTRACTS as GENERIC_CONTRACTS
from .logic import CONTRACTS as LOGIC_CONTRACTS
from .plate_reader import CONTRACTS as PLATE_READER_CONTRACTS
from .response_window import CONTRACTS as RESPONSE_WINDOW_CONTRACTS

BUILTIN_CONTRACTS = (
    *GENERIC_CONTRACTS,
    *PLATE_READER_CONTRACTS,
    *RESPONSE_WINDOW_CONTRACTS,
    *LOGIC_CONTRACTS,
    *CYTOMETRY_CONTRACTS,
)


@cache
def builtin_contract_catalog() -> ContractCatalog:
    return ContractCatalog.from_contracts(BUILTIN_CONTRACTS)


__all__ = [
    "BUILTIN_CONTRACTS",
    "CYTOMETRY_CONTRACTS",
    "GENERIC_CONTRACTS",
    "LOGIC_CONTRACTS",
    "PLATE_READER_CONTRACTS",
    "RESPONSE_WINDOW_CONTRACTS",
    "builtin_contract_catalog",
]
