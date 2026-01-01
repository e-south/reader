"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/contracts.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.metadata as md
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from reader.core.errors import ContractError

DType = Literal["string", "int", "float", "bool", "category", "datetime"]


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
    id: str
    description: str
    columns: list[ColumnRule]
    unique_keys: list[list[str]]  # optional candidates; [] means "not enforced"
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

    # unique keys (if declared)
    for key in contract.unique_keys:
        if not key:
            continue
        if df.duplicated(subset=key, keep=False).any():
            raise ContractError(f"[{where}] contract {contract.id}: uniqueness violated for key {key}")


# ---------------------- Built-in contracts ----------------------

BUILTIN: dict[str, DataFrameContract] = {}


def _register(c: DataFrameContract) -> None:
    if c.id in BUILTIN:
        raise RuntimeError(f"duplicate contract id {c.id}")
    BUILTIN[c.id] = c


# tidy.v1
_register(
    DataFrameContract(
        id="tidy.v1",
        description="Tidy long table: position,str | time,float | channel,str | value,float",
        columns=[
            ColumnRule("position", "string"),
            ColumnRule("time", "float", nonnegative=True),
            ColumnRule("channel", "string"),
            ColumnRule("value", "float"),
        ],
        unique_keys=[],
    )
)

# tidy+map.v2 (design_id)
_register(
    DataFrameContract(
        id="tidy+map.v2",
        description="Tidy+metadata: tidy.v1 + treatment,str | design_id,str",
        columns=[
            ColumnRule("position", "string"),
            ColumnRule("time", "float", nonnegative=True),
            ColumnRule("channel", "string"),
            ColumnRule("value", "float"),
            ColumnRule("treatment", "string"),
            ColumnRule("design_id", "string"),
        ],
        unique_keys=[],
    )
)

# cyto.events.v1
_register(
    DataFrameContract(
        id="cyto.events.v1",
        description="Event-level flow cytometry table (tidy long format).",
        columns=[
            ColumnRule("sample_id", "string"),
            ColumnRule("label_id", "string", required=False, allow_nan=True),
            ColumnRule("event_index", "int", nonnegative=True),
            ColumnRule("channel", "string"),
            ColumnRule("value", "float"),
            ColumnRule("source_file", "string", required=False, allow_nan=True),
        ],
        unique_keys=[],
    )
)

# sfxi.vec8.v2 (design_id)
_register(
    DataFrameContract(
        id="sfxi.vec8.v2",
        description="Per design vec8 table with logic shape and anchor-normalized intensity",
        columns=[
            ColumnRule("design_id", "string"),
            ColumnRule("sequence", "string", required=False, allow_nan=True),
            ColumnRule("id", "string", required=False, allow_nan=True),
            ColumnRule("r_logic", "float", nonnegative=True),
            ColumnRule("v00", "float"),
            ColumnRule("v10", "float"),
            ColumnRule("v01", "float"),
            ColumnRule("v11", "float"),
            ColumnRule("y00_star", "float"),
            ColumnRule("y10_star", "float"),
            ColumnRule("y01_star", "float"),
            ColumnRule("y11_star", "float"),
            ColumnRule("flat_logic", "bool"),
        ],
        unique_keys=[["design_id"]],
    )
)

# fold_change.v1
_register(
    DataFrameContract(
        id="fold_change.v1",
        description="Fold-change summary table per (group..., treatment, time, target).",
        columns=[
            ColumnRule("target", "string"),
            ColumnRule("time", "float", nonnegative=True),
            ColumnRule("time_selected", "float", required=False, allow_nan=True),
            ColumnRule("treatment", "string"),
            ColumnRule("FC", "float", required=True, allow_nan=True),  # default name; validated even if NaN
            ColumnRule("log2FC", "float", required=True, allow_nan=True),
            ColumnRule("n", "int", required=True, allow_nan=False),
            ColumnRule("baseline_value", "string", required=True, allow_nan=True),
            ColumnRule("baseline_n", "int", required=True, allow_nan=True),
            ColumnRule("baseline_time", "float", required=True, allow_nan=True),
            # common group keys (optional if present)
            ColumnRule("design_id", "string", required=False, allow_nan=True),
        ],
        unique_keys=[],  # allow multiple rows per (group,treatment) across report_times or repeats
    )
)


_register(
    DataFrameContract(
        id="logic_symmetry.v1",
        description="Logic-symmetry per design summary (points + metrics + encodings).",
        columns=[
            # optional design-by columns (keep flexible)
            ColumnRule("design_id", "string", required=False, allow_nan=True),
            ColumnRule("strain", "string", required=False, allow_nan=True),
            ColumnRule("design", "string", required=False, allow_nan=True),
            ColumnRule("construct", "string", required=False, allow_nan=True),
            # replicate counts per corner
            ColumnRule("n00", "int"),
            ColumnRule("n10", "int"),
            ColumnRule("n01", "int"),
            ColumnRule("n11", "int"),
            # means per corner
            ColumnRule("b00", "float"),
            ColumnRule("b10", "float"),
            ColumnRule("b01", "float"),
            ColumnRule("b11", "float"),
            # sds per corner
            ColumnRule("sd00", "float"),
            ColumnRule("sd10", "float"),
            ColumnRule("sd01", "float"),
            ColumnRule("sd11", "float"),
            # metrics
            ColumnRule("r", "float"),
            ColumnRule("log_r", "float"),
            ColumnRule("cv", "float"),
            ColumnRule("u00", "float"),
            ColumnRule("u10", "float"),
            ColumnRule("u01", "float"),
            ColumnRule("u11", "float"),
            ColumnRule("L", "float"),
            ColumnRule("A", "float"),
            # baseline and encodings
            ColumnRule("baseline_corner", "string"),
            ColumnRule("baseline_value", "float"),
            ColumnRule("size_value", "float", required=False, allow_nan=True),
            ColumnRule("hue_value", "string", required=False, allow_nan=True),
            ColumnRule("alpha_value", "float", required=False, allow_nan=True),
            ColumnRule("shape_value", "string", required=False, allow_nan=True),
        ],
        unique_keys=[],
    )
)


def _load_external_contracts() -> None:
    """
    Load optional DataFrameContract definitions from entry points.

    Entry point group: reader.contracts
    Each entry point may return:
      • a DataFrameContract
      • an iterable of DataFrameContract
      • a callable that returns one of the above
    """
    try:
        eps = md.entry_points(group="reader.contracts")
    except Exception:
        return
    for ep in eps:
        obj = ep.load()
        if callable(obj):
            obj = obj()
        if isinstance(obj, DataFrameContract):
            _register(obj)
            continue
        if isinstance(obj, list | tuple | set):
            for c in obj:
                if not isinstance(c, DataFrameContract):
                    raise RuntimeError(f"reader.contracts entry {ep.name} returned non-contract: {type(c)}")
                _register(c)
            continue
        raise RuntimeError(f"reader.contracts entry {ep.name} returned invalid object: {type(obj)}")


_load_external_contracts()
