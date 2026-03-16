"""
--------------------------------------------------------------------------------
<reader project>
src/reader/contracts/catalog.py

Explicit dataframe-contract catalog and contract-surface helpers.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import pandas as pd

from reader.core.errors import ContractError

from .model import ContractId, ContractToken, DataFrameContract, validate_df


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
        self._contracts = dict(contracts)

    @classmethod
    def from_contracts(cls, contracts: Iterable[DataFrameContract]) -> ContractCatalog:
        normalized: dict[ContractId, DataFrameContract] = {}
        for contract in contracts:
            if contract.id in normalized:
                raise ContractError(f"duplicate contract id {contract.id!r}")
            normalized[contract.id] = contract
        for contract in normalized.values():
            for parent in contract.parents:
                if parent == contract.id:
                    raise ContractError(f"contract {contract.id!r} cannot list itself as a parent")
                if parent not in normalized:
                    raise ContractError(f"contract {contract.id!r} references unknown parent {parent!r}")
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
