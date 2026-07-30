from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from reader_workbench.contracts import ContractCatalog
from reader_workbench.errors import ContractError, ExecutionError
from reader_workbench.workbench.graph import OutputRef
from reader_workbench.workbench.ports import InputPortSpec, OutputPortSpec
from reader_workbench.workbench.records import PathDescription, SourceRecordCollection
from reader_workbench.workbench.registry import Plugin


def _assert_input_ports(
    plugin: Plugin,
    inputs: dict[str, Any],
    *,
    contracts: ContractCatalog,
    where: str,
) -> None:
    declared = plugin.input_ports()
    allowed = set(declared)
    for name, port in declared.items():
        if name not in inputs:
            if port.optional:
                continue
            raise ExecutionError(f"[{where}] input '{name}' is required by plugin but not provided in 'reads'")
        value = inputs[name]
        _assert_input_port_value(
            name=name,
            port=port,
            value=value,
            contracts=contracts,
            where=where,
        )
    extra = sorted(set(inputs) - allowed)
    if extra:
        raise ExecutionError(f"[{where}] unexpected inputs provided: {extra} (allowed: {sorted(allowed)})")


def _assert_output_ports(
    output_ports: Mapping[str, OutputPortSpec],
    outputs: dict[str, Any],
    *,
    contracts: ContractCatalog,
    where: str,
) -> None:
    if set(outputs) != set(output_ports):
        raise ExecutionError(f"[{where}] plugin must emit outputs {sorted(output_ports)} but emitted {sorted(outputs)}")
    for name, port in output_ports.items():
        value = outputs[name]
        if port.kind == "dataframe":
            if not isinstance(value, pd.DataFrame):
                raise ExecutionError(
                    f"[{where}] output '{name}' must be a DataFrame for dataframe port kind, got {type(value).__name__}"
                )
            try:
                contracts.validate(value, contract_id=port.contract or "", where=where)
            except ContractError as err:
                raise ExecutionError(str(err)) from err
            continue
        if isinstance(value, pd.DataFrame):
            msg = f"[{where}] output '{name}' is declared as {port.kind} but returned a DataFrame"
            raise ExecutionError(msg)
        _coerce_file_output(port=port, value=value, where=where, name=name)


def _resolve_runtime_output_ports(
    plugin: Plugin,
    *,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    cfg: Any,
    contracts: ContractCatalog,
    where: str,
) -> dict[str, OutputPortSpec]:
    declared = dict(plugin.output_ports())
    resolved = dict(plugin.resolve_output_ports(inputs=inputs, outputs=outputs, cfg=cfg, where=where))
    if set(resolved) != set(declared):
        raise ExecutionError(
            f"[{where}] runtime output ports must match declared outputs: "
            f"declared={sorted(declared)} resolved={sorted(resolved)}"
        )
    for name, declared_port in declared.items():
        resolved_port = resolved[name]
        if resolved_port.kind != declared_port.kind:
            raise ExecutionError(
                f"[{where}] runtime output port '{name}' changed kind from {declared_port.kind!r} "
                f"to {resolved_port.kind!r}"
            )
        if declared_port.kind != "dataframe":
            if resolved_port.contract is not None:
                raise ExecutionError(
                    f"[{where}] runtime output port '{name}' of kind {declared_port.kind!r} "
                    "must not resolve a dataframe contract"
                )
            continue
        if declared_port.contract is None or resolved_port.contract is None:
            raise ExecutionError(f"[{where}] dataframe output port '{name}' must resolve a non-empty contract id")
        try:
            if contracts.satisfies(actual=resolved_port.contract, expected=declared_port.contract):
                continue
        except ContractError as err:
            raise ExecutionError(str(err)) from err
        raise ExecutionError(
            f"[{where}] runtime contract for output '{name}' must satisfy declared contract "
            f"{declared_port.contract!r}, got {resolved_port.contract!r}"
        )
    return resolved


def _resolve_output_labels(
    *,
    step_id: str,
    output_ports: Mapping[str, OutputPortSpec],
    writes: dict[str, OutputRef],
) -> dict[str, OutputRef]:
    unknown = sorted(set(writes) - set(output_ports))
    if unknown:
        raise ExecutionError(
            f"[{step_id}] writes includes unknown outputs: {unknown} (expected: {sorted(output_ports)})"
        )
    labels: dict[str, OutputRef] = {}
    for out_name, port in output_ports.items():
        if port.kind != "dataframe":
            if out_name in writes:
                raise ExecutionError(
                    f"[{step_id}] writes cannot target output '{out_name}' (port kind is {port.kind!r})."
                )
            continue
        label = writes[out_name] if out_name in writes else OutputRef(record_id=f"{step_id}/{out_name}")
        if not isinstance(label, OutputRef) or not label.record_id.strip():
            raise ExecutionError(f"[{step_id}] writes for '{out_name}' must be a non-empty record ref.")
        labels[out_name] = label
    if len({ref.record_id for ref in labels.values()}) != len(labels):
        rendered = {name: ref.record_id for name, ref in labels.items()}
        raise ExecutionError(f"[{step_id}] writes produce duplicate output labels: {rendered}")
    return labels


def collect_file_output_paths(
    *,
    output_ports: Mapping[str, OutputPortSpec],
    outputs: Mapping[str, Any],
    where: str,
) -> list[Path]:
    collected: list[Path] = []
    for name, port in output_ports.items():
        if port.kind == "dataframe":
            continue
        collected.extend(_coerce_file_output(port=port, value=outputs[name], where=where, name=name))
    return collected


def collect_file_output_descriptions(
    *,
    output_ports: Mapping[str, OutputPortSpec],
    outputs: Mapping[str, Any],
) -> list[PathDescription]:
    descriptions: list[PathDescription] = []
    for name, port in output_ports.items():
        if port.kind == "dataframe":
            continue
        value = outputs[name]
        if isinstance(value, PathDescription):
            descriptions.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            descriptions.extend(item for item in value if isinstance(item, PathDescription))
    return descriptions


def _assert_input_port_value(
    *,
    name: str,
    port: InputPortSpec,
    value: Any,
    contracts: ContractCatalog,
    where: str,
) -> None:
    if port.kind == "dataframe":
        actual_contract = getattr(value, "contract_id", None)
        if actual_contract is None:
            raise ExecutionError(
                f"[{where}] input '{name}' expects a dataframe artifact but got {type(value).__name__}"
            )
        if port.contract is None:
            return
        try:
            if contracts.satisfies(actual=actual_contract, expected=port.contract):
                return
        except ContractError as err:
            msg = str(err)
        else:
            msg = f"[{where}] input '{name}' must be contract {port.contract} but got {actual_contract}"
        raise ExecutionError(msg)
    if port.kind == "record_collection":
        if not isinstance(value, SourceRecordCollection):
            raise ExecutionError(
                f"[{where}] input '{name}' expects a source record collection but got {type(value).__name__}"
            )
        for item in value:
            actual_contract = getattr(item.record, "contract_id", None)
            if actual_contract is None:
                raise ExecutionError(
                    f"[{where}] input '{name}' source {item.ref.resource_id!r} is not a dataframe artifact"
                )
            if port.contract is None:
                continue
            try:
                compatible = contracts.satisfies(actual=actual_contract, expected=port.contract)
            except ContractError as err:
                raise ExecutionError(str(err)) from err
            if not compatible:
                raise ExecutionError(
                    f"[{where}] input '{name}' source {item.ref.resource_id!r} must satisfy "
                    f"contract {port.contract!r} but got {actual_contract!r}"
                )
        return
    if port.kind == "file_path":
        if isinstance(value, Path):
            return
        raise ExecutionError(f"[{where}] input '{name}' expects a file path but got {type(value).__name__}")
    if port.kind == "file_set":
        if isinstance(value, tuple) and value and all(isinstance(item, Path) for item in value):
            return
        raise ExecutionError(
            f"[{where}] input '{name}' expects a non-empty tuple of file paths but got {type(value).__name__}"
        )
    if port.kind == "file_bundle":
        files = getattr(value, "files", None)
        if files is None:
            raise ExecutionError(f"[{where}] input '{name}' expects a file bundle but got {type(value).__name__}")
        return
    raise ExecutionError(f"[{where}] input '{name}' uses unknown port kind {port.kind!r}")


def _coerce_file_output(
    *,
    port: OutputPortSpec,
    value: Any,
    where: str,
    name: str,
) -> list[Path]:
    if port.kind == "file_path":
        if isinstance(value, PathDescription):
            return [value.path]
        if isinstance(value, str | Path):
            return [_validated_output_path(value, where=where, name=name)]
        raise ExecutionError(f"[{where}] output '{name}' must be a path-like value for file_path ports")
    if port.kind == "file_bundle":
        if isinstance(value, PathDescription):
            return [value.path]
        if isinstance(value, str | Path):
            return [_validated_output_path(value, where=where, name=name)]
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if not value:
                raise ExecutionError(f"[{where}] output '{name}' file bundle must contain at least one file")
            paths: list[Path] = []
            for item in value:
                if isinstance(item, PathDescription):
                    paths.append(item.path)
                    continue
                if not isinstance(item, str | Path):
                    raise ExecutionError(
                        f"[{where}] output '{name}' file bundle entries must be path-like or PathDescription, "
                        f"got {type(item).__name__}"
                    )
                paths.append(_validated_output_path(item, where=where, name=name))
            return paths
        raise ExecutionError(f"[{where}] output '{name}' must be a list of path-like values for file_bundle ports")
    raise ExecutionError(f"[{where}] output '{name}' uses unknown non-dataframe port kind {port.kind!r}")


def _validated_output_path(value: str | Path, *, where: str, name: str) -> Path:
    if isinstance(value, str) and not value.strip():
        raise ExecutionError(f"[{where}] output '{name}' paths must be non-empty")
    path = Path(value)
    if path == Path("."):
        raise ExecutionError(f"[{where}] output '{name}' paths must identify files")
    return path
