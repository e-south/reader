from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from reader.contracts import ContractId, OutputContractSurface
from reader.errors import RegistryError

type PortKind = Literal["dataframe", "file_path", "file_set", "file_bundle"]


@dataclass(frozen=True)
class InputPortSpec:
    name: str
    kind: PortKind
    contract: ContractId | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        _validate_port_name(self.name)
        if self.kind == "dataframe":
            _validate_dataframe_contract(self.contract, where=f"Dataframe input port {self.name!r}", optional=True)
            return
        if self.contract is not None:
            raise RegistryError(f"Input port {self.name!r} of kind {self.kind!r} must not declare a dataframe contract")

    def render(self) -> str:
        if self.kind == "dataframe":
            return self.contract or "dataframe"
        if self.kind == "file_path":
            return "file"
        if self.kind == "file_set":
            return "file set"
        return "file bundle"


@dataclass(frozen=True)
class OutputPortSpec:
    name: str
    kind: PortKind
    contract: ContractId | None = None
    surface: OutputContractSurface | None = None

    def __post_init__(self) -> None:
        _validate_port_name(self.name)
        if self.kind == "file_set":
            raise RegistryError("Output ports must not use input-only kind 'file_set'")
        if self.kind == "dataframe":
            _validate_dataframe_contract(self.contract, where=f"Dataframe output port {self.name!r}", optional=False)
            return
        if self.contract is not None:
            raise RegistryError(
                f"Output port {self.name!r} of kind {self.kind!r} must not declare a dataframe contract"
            )
        if self.surface is not None:
            raise RegistryError(f"Non-dataframe output port {self.name!r} must not declare a contract surface")

    @property
    def contract_surface(self) -> OutputContractSurface | None:
        if self.kind != "dataframe":
            return None
        if self.surface is not None:
            return self.surface
        return OutputContractSurface(minimum=self.contract or "")

    def render(self) -> str:
        if self.kind == "dataframe":
            surface = self.contract_surface
            if surface is None:
                raise RegistryError(f"Dataframe output port {self.name!r} is missing a contract surface")
            return surface.render()
        if self.kind == "file_path":
            return "file"
        return "file bundle"


def dataframe_input(name: str, contract: ContractId | None = None, *, optional: bool = False) -> InputPortSpec:
    return InputPortSpec(name=name, kind="dataframe", contract=contract, optional=optional)


def file_path_input(name: str, *, optional: bool = False) -> InputPortSpec:
    return InputPortSpec(name=name, kind="file_path", optional=optional)


def file_set_input(name: str, *, optional: bool = False) -> InputPortSpec:
    return InputPortSpec(name=name, kind="file_set", optional=optional)


def file_bundle_input(name: str, *, optional: bool = False) -> InputPortSpec:
    return InputPortSpec(name=name, kind="file_bundle", optional=optional)


def dataframe_output(
    name: str,
    contract: ContractId,
    *,
    surface: OutputContractSurface | None = None,
) -> OutputPortSpec:
    return OutputPortSpec(name=name, kind="dataframe", contract=contract, surface=surface)


def file_path_output(name: str) -> OutputPortSpec:
    return OutputPortSpec(name=name, kind="file_path")


def file_bundle_output(name: str) -> OutputPortSpec:
    return OutputPortSpec(name=name, kind="file_bundle")


def validate_input_ports(
    ports: Mapping[str, InputPortSpec],
    *,
    where: str,
) -> dict[str, InputPortSpec]:
    normalized: dict[str, InputPortSpec] = {}
    for key, port in ports.items():
        if not isinstance(port, InputPortSpec):
            raise RegistryError(f"{where}: input port {key!r} must be an InputPortSpec")
        if key != port.name:
            raise RegistryError(f"{where}: input port key {key!r} must match declared name {port.name!r}")
        if key in normalized:
            raise RegistryError(f"{where}: duplicate input port {key!r}")
        normalized[key] = port
    return normalized


def validate_output_ports(
    ports: Mapping[str, OutputPortSpec],
    *,
    where: str,
) -> dict[str, OutputPortSpec]:
    normalized: dict[str, OutputPortSpec] = {}
    for key, port in ports.items():
        if not isinstance(port, OutputPortSpec):
            raise RegistryError(f"{where}: output port {key!r} must be an OutputPortSpec")
        if key != port.name:
            raise RegistryError(f"{where}: output port key {key!r} must match declared name {port.name!r}")
        if key in normalized:
            raise RegistryError(f"{where}: duplicate output port {key!r}")
        normalized[key] = port
    return normalized


def _validate_port_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise RegistryError("Port names must be non-empty strings")
    if name.endswith("?"):
        raise RegistryError(f"Port name {name!r} must not encode optionality with a '?' suffix")
    if name == "files":
        raise RegistryError("Port name 'files' is reserved; use a typed file_path, file_set, or file_bundle port")


def _validate_dataframe_contract(contract: ContractId | None, *, where: str, optional: bool) -> None:
    if contract is None and optional:
        return
    if not isinstance(contract, str) or not contract.strip():
        requirement = "must be a non-empty contract id" if optional else "must declare a non-empty contract id"
        raise RegistryError(f"{where} {requirement}")
    if contract in {"none", "files"}:
        raise RegistryError(f"{where} uses reserved contract id {contract!r}")
