from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reader_workbench.contracts import ContractCatalog
    from reader_workbench.protocols.model import BoundProtocol, ProtocolBinding, ProtocolCatalog
    from reader_workbench.workbench.records.store import RecordStore
    from reader_workbench.workbench.registry import Registry


@dataclass(frozen=True)
class ReaderRuntime:
    """Single composed runtime for contracts, protocols, plugins, and record stores."""

    contracts: ContractCatalog
    protocols: ProtocolCatalog
    plugins: Registry

    def bind_protocol(self, binding: ProtocolBinding) -> BoundProtocol:
        return self.protocols.bind(binding)

    def record_store(
        self,
        outputs_dir: Path,
        *,
        plots_subdir: str | None = "plots",
        exports_subdir: str | None = "exports",
        experiment_root: Path | None = None,
        create: bool = True,
    ) -> RecordStore:
        return import_module("reader_workbench.workbench.records.store").RecordStore(
            outputs_dir,
            contracts=self.contracts,
            plots_subdir=plots_subdir,
            exports_subdir=exports_subdir,
            experiment_root=experiment_root,
            create=create,
        )
