"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/context.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    # Only imported for type checking; never executed at runtime.
    from reader.lib.microplates.style import PaletteBook


@dataclass(frozen=True)
class RunContext:
    exp_dir: Path
    outputs_dir: Path
    artifacts_dir: Path
    plots_dir: Path
    manifest_path: Path
    logger: logging.Logger
    palette_book: Optional[PaletteBook]
    strict: bool = True
    collections: Mapping[str, Any] | None = None

