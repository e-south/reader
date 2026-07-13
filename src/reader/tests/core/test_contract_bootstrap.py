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
    assert {
        "tidy.v1",
        "plate_reader.annotated.v1",
        "plate_reader.response_window.wells.v2",
        "plate_reader.response_window.designs.v2",
        "plate_reader.response_window.bootstrap_draws.v2",
        "plate_reader.response_window.traces.v2",
        "plate_reader.response_window.events.v2",
        "fold_change.v1",
        "plate_reader.sponge_trace.v1",
        "plate_reader.sponge_summary.v1",
        "sfxi.vec8.v2",
        "sfxi.vec8.v3",
        "cytometer.channels.v1",
    } <= set(catalog.ids())
    vec8_v2 = catalog.require("sfxi.vec8.v2")
    vec8_v3 = catalog.require("sfxi.vec8.v3")
    delta_v2 = next(column for column in vec8_v2.columns if column.name == "intensity_log2_offset_delta")
    delta_v3 = next(column for column in vec8_v3.columns if column.name == "intensity_log2_offset_delta")
    assert delta_v2.required is False
    assert delta_v3.required is True
    assert builtin_contract_catalog() is catalog
