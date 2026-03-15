from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd

from reader.core.contracts import BUILTIN, contract_satisfies, validate_df
from reader.core.errors import ContractError, ExecutionError
from reader.core.registry import Plugin


def _assert_input_contracts(
    plugin: Plugin,
    inputs: dict[str, Any],
    *,
    where: str,
    strict: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    required_inputs = plugin.input_contracts()
    allowed: set[str] = set()
    for raw_name, contract_id in required_inputs.items():
        optional = raw_name.endswith("?")
        name = raw_name[:-1] if optional else raw_name
        allowed.add(name)
        if name not in inputs:
            if optional:
                continue
            raise ExecutionError(f"[{where}] input '{name}' is required by plugin but not provided in 'reads'")
        if contract_id == "none":
            continue
        inp = inputs[name]
        actual_contract = getattr(inp, "contract_id", None)
        if contract_satisfies(actual=actual_contract, expected=contract_id):
            continue
        msg = f"[{where}] input '{name}' must be contract {contract_id} but got {actual_contract}"
        if strict:
            raise ExecutionError(msg)
        if logger is not None:
            logger.warning("contract relaxed • %s", msg)
        else:
            warnings.warn(msg, stacklevel=2)
    extra = sorted(set(inputs) - allowed)
    if extra:
        raise ExecutionError(f"[{where}] unexpected inputs provided: {extra} (allowed: {sorted(allowed)})")


def _assert_output_contracts(
    expected_contracts: Mapping[str, str],
    outputs: dict[str, Any],
    *,
    where: str,
    strict: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    if set(outputs) != set(expected_contracts):
        raise ExecutionError(
            f"[{where}] plugin must emit outputs {sorted(expected_contracts)} but emitted {sorted(outputs)}"
        )
    for name, contract_id in expected_contracts.items():
        value = outputs[name]
        if contract_id == "none":
            if isinstance(value, pd.DataFrame):
                msg = f"[{where}] output '{name}' is declared as contract 'none' but returned a DataFrame"
                if strict:
                    raise ExecutionError(msg)
                if logger is not None:
                    logger.warning("contract relaxed • %s", msg)
                else:
                    warnings.warn(msg, stacklevel=2)
            continue
        if not isinstance(value, pd.DataFrame):
            continue
        contract = BUILTIN.get(contract_id)
        if contract is None:
            msg = f"[{where}] unknown contract id {contract_id}"
            if strict:
                raise ExecutionError(msg)
            if logger is not None:
                logger.warning("contract relaxed • %s", msg)
            else:
                warnings.warn(msg, stacklevel=2)
            continue
        try:
            validate_df(value, contract, where=where)
        except ContractError as err:
            msg = str(err)
            if strict:
                raise ExecutionError(msg) from err
            if logger is not None:
                logger.warning("contract relaxed • %s", msg)
            else:
                warnings.warn(msg, stacklevel=2)


def _resolve_runtime_output_contracts(
    plugin: Plugin,
    *,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    cfg: Any,
    where: str,
) -> dict[str, str]:
    declared = dict(plugin.output_contracts())
    resolved = dict(plugin.resolve_output_contracts(inputs=inputs, outputs=outputs, cfg=cfg, where=where))
    if set(resolved) != set(declared):
        raise ExecutionError(
            f"[{where}] runtime output contracts must match declared outputs: "
            f"declared={sorted(declared)} resolved={sorted(resolved)}"
        )
    for name, declared_id in declared.items():
        resolved_id = resolved[name]
        if declared_id == "none":
            if resolved_id != "none":
                raise ExecutionError(
                    f"[{where}] output '{name}' is declared as 'none' but resolved to contract {resolved_id!r}"
                )
            continue
        if contract_satisfies(actual=resolved_id, expected=declared_id):
            continue
        raise ExecutionError(
            f"[{where}] runtime contract for output '{name}' must satisfy declared contract "
            f"{declared_id!r}, got {resolved_id!r}"
        )
    return resolved


def _resolve_output_labels(*, step_id: str, output_contracts: dict[str, str], writes: dict[str, str]) -> dict[str, str]:
    unknown = sorted(set(writes) - set(output_contracts))
    if unknown:
        raise ExecutionError(
            f"[{step_id}] writes includes unknown outputs: {unknown} (expected: {sorted(output_contracts)})"
        )
    labels: dict[str, str] = {}
    for out_name, contract_id in output_contracts.items():
        if out_name in writes:
            if contract_id == "none":
                raise ExecutionError(f"[{step_id}] writes cannot target output '{out_name}' (contract is 'none').")
            label = writes[out_name]
        else:
            label = f"{step_id}/{out_name}"
        if not isinstance(label, str) or not label.strip():
            raise ExecutionError(f"[{step_id}] writes for '{out_name}' must be a non-empty string.")
        labels[out_name] = label
    if len(set(labels.values())) != len(labels):
        raise ExecutionError(f"[{step_id}] writes produce duplicate output labels: {labels}")
    return labels
