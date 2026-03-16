"""
--------------------------------------------------------------------------------
<reader project>
src/reader/contracts/model.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from reader.core.errors import ContractError

type ContractId = str
DType = Literal["string", "int", "float", "bool", "category", "datetime"]
type ContractToken = ContractId | Literal["none"]


@dataclass(frozen=True)
class ColumnRule:
    name: str
    dtype: DType
    required: bool = True
    allow_nan: bool = False
    monotone_non_decreasing: bool = False
    nonnegative: bool = False
    allowed_values: list[str] | None = None


@dataclass(frozen=True)
class DataFrameContract:
    id: ContractId
    description: str
    columns: list[ColumnRule]
    unique_keys: list[list[str]]
    parents: tuple[ContractId, ...] = ()
    domain: str | None = None
    kind: str | None = None
    primary_index: list[str] | None = None
    notes: str | None = None


def _is_dtype(series: pd.Series, want: DType) -> bool:
    if want == "string":
        return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    if want == "int":
        return pd.api.types.is_integer_dtype(series)
    if want == "float":
        return pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series)
    if want == "bool":
        return pd.api.types.is_bool_dtype(series)
    if want == "category":
        return pd.api.types.is_categorical_dtype(series)
    if want == "datetime":
        return pd.api.types.is_datetime64_any_dtype(series)
    return False


def validate_df(df: pd.DataFrame, contract: DataFrameContract, *, where: str) -> None:
    """Assert df matches the contract exactly; raise ContractError on first failure."""
    cols = set(df.columns)

    for rule in contract.columns:
        if rule.required and rule.name not in cols:
            raise ContractError(f"[{where}] contract {contract.id}: missing required column '{rule.name}'")

    for rule in contract.columns:
        if rule.name not in cols:
            continue
        s = df[rule.name]
        if not _is_dtype(s, rule.dtype):
            raise ContractError(
                f"[{where}] contract {contract.id}: column '{rule.name}' has dtype {s.dtype} but expected {rule.dtype}"
            )
        if not rule.allow_nan and s.isna().any():
            raise ContractError(
                f"[{where}] contract {contract.id}: column '{rule.name}' contains NaN but allow_nan=false"
            )
        if rule.nonnegative and (pd.to_numeric(s, errors="coerce") < 0).any():
            raise ContractError(f"[{where}] contract {contract.id}: column '{rule.name}' must be nonnegative")
        if rule.allowed_values is not None:
            bad = sorted(set(map(str, s.dropna().astype(str))) - set(map(str, rule.allowed_values)))
            if bad:
                raise ContractError(
                    f"[{where}] contract {contract.id}: column '{rule.name}' contains values outside allowed set: {bad[:5]}"
                )

    for key in contract.unique_keys:
        if not key:
            continue
        if df.duplicated(subset=key, keep=False).any():
            raise ContractError(f"[{where}] contract {contract.id}: uniqueness violated for key {key}")
