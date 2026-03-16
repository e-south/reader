from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reader.contracts import ContractCatalog
from reader.workbench.assets.types import AssetCatalog
from reader.workbench.records.store import RecordStore
from reader.workbench.registry import Registry


@dataclass(frozen=True)
class ReaderRuntime:
    """Single composed runtime for contracts, plugins, assets, and record stores."""

    contracts: ContractCatalog
    plugins: Registry
    assets: AssetCatalog

    def record_store(
        self,
        outputs_dir: Path,
        *,
        plots_subdir: str | None = "plots",
        exports_subdir: str | None = "exports",
        create: bool = True,
    ) -> RecordStore:
        return RecordStore(
            outputs_dir,
            contracts=self.contracts,
            plots_subdir=plots_subdir,
            exports_subdir=exports_subdir,
            create=create,
        )
