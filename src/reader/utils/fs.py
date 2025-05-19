"""
--------------------------------------------------------------------------------
<reader project>
src/reader/utils/fs.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import os
from pathlib import Path


def ensure_dir(path: str | Path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
