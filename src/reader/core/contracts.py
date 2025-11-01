"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/contracts.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import pandas as pd

from reader.core.errors import ContractError

DType = Literal["string","int","float","bool","category","datetime"]

@dataclass(frozen=True)
class ColumnRule:
    name: str
    dtype: DType
    required: bool = True
    allow_nan: bool = False
    monotone_non_decreasing: bool = False
    nonnegative: bool = False
    allowed_values: Optional[List[str]] = None

@dataclass(frozen=True)
class DataFrameContract:
    id: str
    description: str
    columns: List[ColumnRule]
    unique_keys: List[List[str]]  # optional candidates; [] means "not enforced"
    primary_index: Optional[List[str]] = None
    notes: Optional[str] = None


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

    # required columns
    for rule in contract.columns:
        if rule.required and rule.name not in cols:
            raise ContractError(f"[{where}] contract {contract.id}: missing required column '{rule.name}'")

    # column dtypes & invariants
    for rule in contract.columns:
        if rule.name not in cols:
            continue
        s = df[rule.name]
        if not _is_dtype(s, rule.dtype):
            raise ContractError(f"[{where}] contract {contract.id}: column '{rule.name}' has dtype {s.dtype} but expected {rule.dtype}")
        if not rule.allow_nan and s.isna().any():
            raise ContractError(f"[{where}] contract {contract.id}: column '{rule.name}' contains NaN but allow_nan=false")
        if rule.nonnegative and (pd.to_numeric(s, errors="coerce") < 0).any():
            raise ContractError(f"[{where}] contract {contract.id}: column '{rule.name}' must be nonnegative")
        if rule.allowed_values is not None:
            bad = sorted(set(map(str, s.dropna().astype(str))) - set(map(str, rule.allowed_values)))
            if bad:
                raise ContractError(f"[{where}] contract {contract.id}: column '{rule.name}' contains values outside allowed set: {bad[:5]}")

    # unique keys (if declared)
    for key in contract.unique_keys:
        if not key:
            continue
        if df.duplicated(subset=key, keep=False).any():
            raise ContractError(f"[{where}] contract {contract.id}: uniqueness violated for key {key}")


# ---------------------- Built-in contracts ----------------------

BUILTIN: Dict[str, DataFrameContract] = {}

def _register(c: DataFrameContract) -> None:
    if c.id in BUILTIN:
        raise RuntimeError(f"duplicate contract id {c.id}")
    BUILTIN[c.id] = c

# tidy.v1
_register(DataFrameContract(
    id="tidy.v1",
    description="Tidy long table: position,str | time,float | channel,str | value,float",
    columns=[
        ColumnRule("position","string"),
        ColumnRule("time","float", nonnegative=True),
        ColumnRule("channel","string"),
        ColumnRule("value","float"),
    ],
    unique_keys=[],
))

# tidy+map.v1
_register(DataFrameContract(
    id="tidy+map.v1",
    description="Tidy+metadata: tidy.v1 + treatment,str | genotype,str | batch,int",
    columns=[
        ColumnRule("position","string"),
        ColumnRule("time","float", nonnegative=True),
        ColumnRule("channel","string"),
        ColumnRule("value","float"),
        ColumnRule("treatment","string"),
        ColumnRule("genotype","string"),
        ColumnRule("batch","int"),
    ],
    unique_keys=[],
))

# sfxi.vec8.v1
_register(DataFrameContract(
    id="sfxi.vec8.v1",
    description="Per design×batch vec8 table with logic shape and anchor-normalized intensity",
    columns=[
        ColumnRule("genotype","string"),
        ColumnRule("sequence","string", required=False, allow_nan=True),
        ColumnRule("r_logic","float", nonnegative=True),
        ColumnRule("v00","float"), ColumnRule("v10","float"),
        ColumnRule("v01","float"), ColumnRule("v11","float"),
        ColumnRule("y00_star","float"), ColumnRule("y10_star","float"),
        ColumnRule("y01_star","float"), ColumnRule("y11_star","float"),
        ColumnRule("flat_logic","bool"),
    ],
    unique_keys=[["genotype"]],
))

# fold_change.v1
_register(DataFrameContract(
    id="fold_change.v1",
    description="Fold-change summary table per (group..., treatment, time, target).",
    columns=[
        ColumnRule("target", "string"),
        ColumnRule("time", "float", nonnegative=True),
        ColumnRule("treatment", "string"),
        ColumnRule("FC", "float", required=True, allow_nan=True),         # default name; validated even if NaN
        ColumnRule("log2FC", "float", required=True, allow_nan=True),
        ColumnRule("n", "int", required=True, allow_nan=False),
        ColumnRule("baseline_value", "string", required=True, allow_nan=True),
        ColumnRule("baseline_n", "int", required=True, allow_nan=True),
        ColumnRule("baseline_time", "float", required=True, allow_nan=True),
        # common group keys (optional if present)
        ColumnRule("genotype", "string", required=False, allow_nan=True),
        ColumnRule("batch", "int", required=False, allow_nan=True),
    ],
    unique_keys=[],   # allow multiple rows per (group,treatment) across report_times or repeats
))
