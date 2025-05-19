"""
--------------------------------------------------------------------------------
<reader project>

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path


def setup_logging(output_dir: str | Path) -> None:
    """Configure root logger to write to file and stdout."""
    log_path = Path(output_dir) / 'reader.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
