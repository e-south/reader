"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/core/test_contract_bootstrap.py

Regression coverage for the explicit built-in contract catalog.
--------------------------------------------------------------------------------
"""

from reader.contracts import BUILTIN_CONTRACTS, builtin_contract_catalog


def test_builtin_contract_catalog_is_explicit_and_stable() -> None:
    catalog = builtin_contract_catalog()

    assert BUILTIN_CONTRACTS
    assert {"tidy.v1", "plate_reader.annotated.v1", "fold_change.v1", "sfxi.vec8.v1", "cytometer.channels.v1"} <= set(
        catalog.ids()
    )
    assert builtin_contract_catalog() is catalog
