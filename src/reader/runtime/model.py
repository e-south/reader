from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reader.contracts import ContractCatalog
    from reader.protocols.model import BoundProtocol, ProtocolBinding, ProtocolCatalog
    from reader.workbench.assets.types import AssetCatalog
    from reader.workbench.records.store import RecordStore
    from reader.workbench.registry import Registry


@dataclass(frozen=True)
class ReaderRuntime:
    """Single composed runtime for contracts, protocols, plugins, assets, and record stores."""

    contracts: ContractCatalog
    protocols: ProtocolCatalog
    plugins: Registry
    assets: AssetCatalog

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
        return import_module("reader.workbench.records.store").RecordStore(
            outputs_dir,
            contracts=self.contracts,
            plots_subdir=plots_subdir,
            exports_subdir=exports_subdir,
            experiment_root=experiment_root,
            create=create,
        )
