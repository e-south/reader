from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reader.plotting.style import PaletteBook
    from reader.protocols import BoundProtocol
    from reader.workbench.experiment import ExperimentSemantics


@dataclass(frozen=True)
class RunContext:
    exp_dir: Path
    outputs_dir: Path
    artifacts_dir: Path
    plots_dir: Path
    exports_dir: Path
    records_path: Path
    logger: logging.Logger
    palette_book: PaletteBook | None
    experiment: ExperimentSemantics | None = None
    protocol: BoundProtocol | None = None
    config_digest: str = ""
