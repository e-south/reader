"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/contracts/registry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .model import DataFrameContract

BUILTIN: dict[str, DataFrameContract] = {}


@dataclass(frozen=True)
class OutputContractSurface:
    minimum: str
    runtime_mode: str = "fixed"  # fixed | promoted | passthrough
    promoted: tuple[str, ...] = ()
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


def register_contract(contract: DataFrameContract) -> None:
    if contract.id in BUILTIN:
        raise RuntimeError(f"duplicate contract id {contract.id}")
    BUILTIN[contract.id] = contract


def iter_contract_lineage(contract_id: str) -> Iterable[str]:
    seen: set[str] = set()
    stack = [contract_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        yield current
        contract = BUILTIN.get(current)
        if contract is None:
            continue
        stack.extend(reversed(contract.parents))


def contract_satisfies(*, actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    return expected in set(iter_contract_lineage(actual))
