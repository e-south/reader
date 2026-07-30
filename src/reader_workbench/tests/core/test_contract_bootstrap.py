"""Regression coverage for the explicit built-in contract catalog."""

from dataclasses import replace

import pandas as pd
import pytest

from reader_workbench.contracts import (
    BUILTIN_CONTRACTS,
    ColumnRule,
    ContractCatalog,
    DataFrameContract,
    builtin_contract_catalog,
)
from reader_workbench.errors import ContractError


def test_builtin_contract_catalog_is_explicit_and_stable() -> None:
    catalog = builtin_contract_catalog()

    assert BUILTIN_CONTRACTS
    assert {
        "tidy.v1",
        "plate_reader.annotated.v1",
        "plate_reader.response_window.wells.v3",
        "plate_reader.response_window.designs.v4",
        "plate_reader.response_window.descriptive_resampling_draws.v3",
        "plate_reader.response_window.traces.v3",
        "plate_reader.response_window.events.v2",
        "fold_change.v1",
        "sfxi.vec8.v3",
        "sfxi.vec8_collection.v1",
        "sfxi.vec8_collection.v2",
        "cytometer.channels.v1",
    } <= set(catalog.ids())
    designs = catalog.require("plate_reader.response_window.designs.v4")
    design_columns = {column.name for column in designs.columns}
    assert {
        "observation_stat",
        "descriptive_resampling_draws",
        "descriptive_interval_mass",
        "positive_floor",
        "allowed_max_interior_gap_h",
        "required_min_observations_per_state",
        "min_observation_count_per_state",
        "max_observed_interior_gap_h",
        "max_pre_observed_interior_gap_h",
    } <= design_columns
    assert not any(
        retired in column for column in design_columns for retired in ("replicate", "bootstrap", "confidence", "_ci_")
    )
    assert "plate_reader.response_window.designs.v3" not in catalog.ids()
    assert "plate_reader.response_window.bootstrap_draws.v2" not in catalog.ids()
    assert "plate_reader.sponge_trace.v1" not in catalog.ids()
    assert "plate_reader.sponge_summary.v1" not in catalog.ids()
    vec8_v3 = catalog.require("sfxi.vec8.v3")
    delta_v3 = next(column for column in vec8_v3.columns if column.name == "intensity_log2_offset_delta")
    assert delta_v3.required is True
    collection_v1 = catalog.require("sfxi.vec8_collection.v1")
    collection_v2 = catalog.require("sfxi.vec8_collection.v2")
    assert "source_record_revision_digest" not in {column.name for column in collection_v1.columns}
    assert "source_record_revision_digest" in {column.name for column in collection_v2.columns}
    assert collection_v2.parents == ("sfxi.vec8_collection.v1",)
    assert builtin_contract_catalog() is catalog


def _contract(
    contract_id: str,
    *,
    columns: list[ColumnRule],
    parents: tuple[str, ...] = (),
    unique_keys: list[list[str]] | None = None,
    primary_index: list[str] | None = None,
) -> DataFrameContract:
    return DataFrameContract(
        id=contract_id,
        description=contract_id,
        columns=columns,
        unique_keys=unique_keys or [],
        parents=parents,
        primary_index=primary_index,
    )


def test_contract_catalog_rejects_two_node_lineage_cycle() -> None:
    first = _contract("first.v1", columns=[ColumnRule("id", "string")], parents=("second.v1",))
    second = _contract("second.v1", columns=[ColumnRule("id", "string")], parents=("first.v1",))

    with pytest.raises(ContractError, match=r"lineage cycle.*first\.v1.*second\.v1.*first\.v1"):
        ContractCatalog.from_contracts((first, second))


def test_contract_catalog_rejects_child_missing_parent_required_column() -> None:
    parent = _contract("parent.v1", columns=[ColumnRule("required_id", "string")])
    child = _contract(
        "child.v1",
        columns=[ColumnRule("other", "string")],
        parents=("parent.v1",),
    )

    with pytest.raises(
        ContractError,
        match=r"child\.v1.*parent\.v1.*required column 'required_id'",
    ):
        ContractCatalog.from_contracts((parent, child))


@pytest.mark.parametrize(
    ("parent_rule", "child_rule", "message"),
    [
        (ColumnRule("value", "float"), ColumnRule("value", "string"), "dtype"),
        (
            ColumnRule("value", "float", allow_nan=False),
            ColumnRule("value", "float", allow_nan=True),
            "null values",
        ),
    ],
)
def test_contract_catalog_rejects_incompatible_parent_column_rules(
    parent_rule: ColumnRule,
    child_rule: ColumnRule,
    message: str,
) -> None:
    parent = _contract("parent.v1", columns=[parent_rule])
    child = _contract("child.v1", columns=[child_rule], parents=("parent.v1",))

    with pytest.raises(ContractError, match=message):
        ContractCatalog.from_contracts((parent, child))


@pytest.mark.parametrize(
    "field",
    [
        "required",
        "allow_nan",
        "monotone_non_decreasing",
        "nonnegative",
        "allow_extra_columns",
    ],
)
def test_contract_catalog_rejects_non_boolean_flags(field: str) -> None:
    contract = _contract("invalid-boolean.v1", columns=[ColumnRule("value", "float")])
    if field == "allow_extra_columns":
        contract = replace(contract, allow_extra_columns="false")  # type: ignore[arg-type]
    else:
        rule = replace(contract.columns[0], **{field: "false"})
        contract = replace(contract, columns=(rule,))

    with pytest.raises(ContractError, match=rf"{field}.*bool"):
        ContractCatalog.from_contracts((contract,))


def test_contract_catalog_rejects_child_that_drops_parent_unique_key() -> None:
    parent = _contract(
        "parent.v1",
        columns=[ColumnRule("id", "string")],
        unique_keys=[["id"]],
    )
    child = _contract(
        "child.v1",
        columns=[ColumnRule("id", "string")],
        parents=("parent.v1",),
    )

    with pytest.raises(ContractError, match="does not preserve unique key"):
        ContractCatalog.from_contracts((parent, child))


def test_contract_catalog_owns_immutable_contract_values() -> None:
    allowed_values = ["a", "b"]
    columns = [ColumnRule("id", "string", allowed_values=allowed_values)]
    unique_keys = [["id"]]
    contract = _contract("example.v1", columns=columns, unique_keys=unique_keys)
    catalog = ContractCatalog.from_contracts((contract,))

    columns.clear()
    allowed_values.append("c")
    unique_keys[0].append("missing")

    registered = catalog.require("example.v1")
    assert tuple(rule.name for rule in registered.columns) == ("id",)
    assert registered.columns[0].allowed_values == ("a", "b")
    assert registered.unique_keys == (("id",),)


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        (
            _contract(
                "bad-key.v1",
                columns=[ColumnRule("id", "string")],
                unique_keys=[["missing"]],
            ),
            "unique key.*unknown column 'missing'",
        ),
        (
            _contract(
                "bad-index.v1",
                columns=[ColumnRule("id", "string")],
                primary_index=["missing"],
            ),
            "primary index.*unknown column 'missing'",
        ),
    ],
)
def test_contract_catalog_rejects_unknown_key_and_index_columns(
    contract: DataFrameContract,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        ContractCatalog.from_contracts((contract,))


def test_contract_catalog_enforces_monotone_column_rules() -> None:
    contract = _contract(
        "monotone.v1",
        columns=[ColumnRule("time", "float", monotone_non_decreasing=True)],
    )
    catalog = ContractCatalog.from_contracts((contract,))

    with pytest.raises(ContractError, match="must be monotone non-decreasing"):
        catalog.validate(
            pd.DataFrame({"time": [2.0, 1.0]}),
            contract_id="monotone.v1",
            where="test",
        )
