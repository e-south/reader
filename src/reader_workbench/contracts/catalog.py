"""Explicit dataframe-contract catalog and contract-surface helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import pandas as pd

from reader_workbench.errors import ContractError

from .model import ColumnRule, ContractId, ContractToken, DataFrameContract, DType, validate_df


def _dtype_satisfies(*, child: DType, parent: DType) -> bool:
    """Return whether every value accepted by a child dtype is accepted by its parent."""

    return child == parent or (child == "int" and parent == "float")


def _require_compatible_rule(
    *,
    child_contract: DataFrameContract,
    parent_contract: DataFrameContract,
    child_rule: ColumnRule,
    parent_rule: ColumnRule,
) -> None:
    prefix = (
        f"contract {child_contract.id!r} is not structurally compatible with parent "
        f"{parent_contract.id!r}: column {parent_rule.name!r}"
    )
    if not _dtype_satisfies(child=child_rule.dtype, parent=parent_rule.dtype):
        raise ContractError(f"{prefix} has dtype {child_rule.dtype!r}, expected a subtype of {parent_rule.dtype!r}")
    if not parent_rule.allow_nan and child_rule.allow_nan:
        raise ContractError(f"{prefix} permits null values forbidden by the parent")
    if parent_rule.monotone_non_decreasing and not child_rule.monotone_non_decreasing:
        raise ContractError(f"{prefix} does not preserve the parent's monotone constraint")
    if parent_rule.nonnegative and not child_rule.nonnegative:
        raise ContractError(f"{prefix} does not preserve the parent's nonnegative constraint")
    if parent_rule.allowed_values is not None:
        if child_rule.allowed_values is None:
            raise ContractError(f"{prefix} does not preserve the parent's allowed-values constraint")
        parent_values = {str(value) for value in parent_rule.allowed_values}
        child_values = {str(value) for value in child_rule.allowed_values}
        if not child_values <= parent_values:
            raise ContractError(f"{prefix} permits values outside the parent's allowed set")


def _require_structural_compatibility(*, child: DataFrameContract, parent: DataFrameContract) -> None:
    child_columns = {rule.name: rule for rule in child.columns}
    parent_columns = {rule.name: rule for rule in parent.columns}

    for name, parent_rule in parent_columns.items():
        child_rule = child_columns.get(name)
        if child_rule is None:
            if parent_rule.required:
                raise ContractError(
                    f"contract {child.id!r} is not structurally compatible with parent {parent.id!r}: "
                    f"missing required column {name!r}"
                )
            if child.allow_extra_columns:
                raise ContractError(
                    f"contract {child.id!r} is not structurally compatible with parent {parent.id!r}: "
                    f"optional parent column {name!r} is unconstrained by the child"
                )
            continue
        if parent_rule.required and not child_rule.required:
            raise ContractError(
                f"contract {child.id!r} is not structurally compatible with parent {parent.id!r}: "
                f"required column {name!r} is optional in the child"
            )
        _require_compatible_rule(
            child_contract=child,
            parent_contract=parent,
            child_rule=child_rule,
            parent_rule=parent_rule,
        )

    if not parent.allow_extra_columns:
        extras = sorted(set(child_columns) - set(parent_columns))
        if child.allow_extra_columns or extras:
            detail = "allows undeclared columns" if child.allow_extra_columns else f"declares extra columns {extras}"
            raise ContractError(
                f"contract {child.id!r} is not structurally compatible with parent {parent.id!r}: {detail}"
            )

    child_keys = [set(key) for key in child.unique_keys if key]
    for parent_key in (set(key) for key in parent.unique_keys if key):
        if not any(child_key <= parent_key for child_key in child_keys):
            raise ContractError(
                f"contract {child.id!r} is not structurally compatible with parent {parent.id!r}: "
                f"does not preserve unique key {sorted(parent_key)}"
            )


def _validate_contract_graph(contracts: Mapping[ContractId, DataFrameContract]) -> None:
    for contract_id, contract in contracts.items():
        if not isinstance(contract, DataFrameContract):
            raise ContractError(f"contract catalog entry {contract_id!r} must be a DataFrameContract")
        if not isinstance(contract.id, str) or not contract.id.strip():
            raise ContractError("contract ids must be non-empty strings")
        if contract.id != contract_id:
            raise ContractError(f"contract catalog key {contract_id!r} does not match contract id {contract.id!r}")
        if any(not isinstance(rule, ColumnRule) for rule in contract.columns):
            raise ContractError(f"contract {contract.id!r} columns must contain ColumnRule values")
        column_names = [rule.name for rule in contract.columns]
        if any(not isinstance(name, str) or not name.strip() for name in column_names):
            raise ContractError(f"contract {contract.id!r} column names must be non-empty strings")
        if len(column_names) != len(set(column_names)):
            raise ContractError(f"contract {contract.id!r} declares duplicate column names")
        if type(contract.allow_extra_columns) is not bool:
            raise ContractError(f"contract {contract.id!r} allow_extra_columns must be bool")
        columns_by_name = {rule.name: rule for rule in contract.columns}
        for rule in contract.columns:
            for field in ("required", "allow_nan", "monotone_non_decreasing", "nonnegative"):
                if type(getattr(rule, field)) is not bool:
                    raise ContractError(f"contract {contract.id!r} column {rule.name!r} {field} must be bool")
            if rule.nonnegative and rule.dtype not in {"int", "float"}:
                raise ContractError(
                    f"contract {contract.id!r} column {rule.name!r} uses nonnegative with non-numeric dtype"
                )
            if rule.monotone_non_decreasing and rule.dtype not in {"int", "float", "datetime"}:
                raise ContractError(
                    f"contract {contract.id!r} column {rule.name!r} uses monotonicity with unordered dtype"
                )
        for key in contract.unique_keys:
            if not key:
                raise ContractError(f"contract {contract.id!r} unique keys must not be empty")
            if len(key) != len(set(key)):
                raise ContractError(f"contract {contract.id!r} unique key {list(key)!r} contains duplicate columns")
            for name in key:
                rule = columns_by_name.get(name)
                if rule is None:
                    raise ContractError(f"contract {contract.id!r} unique key references unknown column {name!r}")
                if not rule.required:
                    raise ContractError(f"contract {contract.id!r} unique key column {name!r} must be required")
        if contract.primary_index is not None:
            if not contract.primary_index:
                raise ContractError(f"contract {contract.id!r} primary index must not be empty")
            if len(contract.primary_index) != len(set(contract.primary_index)):
                raise ContractError(f"contract {contract.id!r} primary index contains duplicate columns")
            for name in contract.primary_index:
                rule = columns_by_name.get(name)
                if rule is None:
                    raise ContractError(f"contract {contract.id!r} primary index references unknown column {name!r}")
                if not rule.required:
                    raise ContractError(f"contract {contract.id!r} primary index column {name!r} must be required")
        for parent in contract.parents:
            if parent == contract.id:
                raise ContractError(f"contract {contract.id!r} cannot list itself as a parent")
            if parent not in contracts:
                raise ContractError(f"contract {contract.id!r} references unknown parent {parent!r}")

    visited: set[ContractId] = set()
    active: set[ContractId] = set()
    path: list[ContractId] = []

    def visit(contract_id: ContractId) -> None:
        if contract_id in active:
            cycle_start = path.index(contract_id)
            cycle = [*path[cycle_start:], contract_id]
            raise ContractError("contract lineage cycle detected: " + " -> ".join(cycle))
        if contract_id in visited:
            return
        active.add(contract_id)
        path.append(contract_id)
        for parent in contracts[contract_id].parents:
            visit(parent)
        path.pop()
        active.remove(contract_id)
        visited.add(contract_id)

    for contract_id in sorted(contracts):
        visit(contract_id)

    for child in contracts.values():
        for parent_id in child.parents:
            _require_structural_compatibility(child=child, parent=contracts[parent_id])


@dataclass(frozen=True)
class OutputContractSurface:
    minimum: ContractToken
    runtime_mode: str = "fixed"  # fixed | promoted | passthrough
    promoted: tuple[ContractId, ...] = ()
    note: str | None = None

    def render(self) -> str:
        if self.minimum == "none":
            return "none"
        parts = [self.minimum]
        if self.runtime_mode == "promoted" and self.promoted:
            parts.append(f"runtime may promote to {', '.join(self.promoted)}")
        elif self.runtime_mode == "passthrough":
            hint = f" (e.g. {', '.join(self.promoted)})" if self.promoted else ""
            parts.append(f"runtime preserves stricter compatible input contracts{hint}")
        if self.note:
            parts.append(self.note)
        return "; ".join(parts)


class ContractCatalog:
    """Immutable contract catalog with explicit lineage and validation helpers."""

    def __init__(self, contracts: Mapping[ContractId, DataFrameContract]) -> None:
        normalized = dict(contracts)
        _validate_contract_graph(normalized)
        self._contracts = normalized

    @classmethod
    def from_contracts(cls, contracts: Iterable[DataFrameContract]) -> ContractCatalog:
        normalized: dict[ContractId, DataFrameContract] = {}
        for contract in contracts:
            if contract.id in normalized:
                raise ContractError(f"duplicate contract id {contract.id!r}")
            normalized[contract.id] = contract
        return cls(normalized)

    def all(self) -> tuple[DataFrameContract, ...]:
        return tuple(self._contracts[key] for key in sorted(self._contracts))

    def ids(self) -> tuple[ContractId, ...]:
        return tuple(sorted(self._contracts))

    def __contains__(self, contract_id: object) -> bool:
        return isinstance(contract_id, str) and contract_id in self._contracts

    def get(self, contract_id: ContractToken | None) -> DataFrameContract | None:
        if contract_id in (None, "none"):
            return None
        return self._contracts.get(contract_id)

    def require(self, contract_id: ContractId) -> DataFrameContract:
        try:
            return self._contracts[contract_id]
        except KeyError as exc:
            raise ContractError(f"unknown contract id {contract_id!r}") from exc

    def iter_lineage(self, contract_id: ContractId) -> Iterator[ContractId]:
        seen: set[ContractId] = set()
        stack: list[ContractId] = [contract_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            contract = self.require(current)
            seen.add(current)
            yield current
            stack.extend(reversed(contract.parents))

    def satisfies(self, *, actual: ContractToken | None, expected: ContractId) -> bool:
        if actual in (None, "none"):
            return False
        return expected in set(self.iter_lineage(actual))

    def validate(self, df: pd.DataFrame, *, contract_id: ContractId, where: str) -> None:
        validate_df(df, self.require(contract_id), where=where)
