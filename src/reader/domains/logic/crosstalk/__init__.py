"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/logic/crosstalk/__init__.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .pairs import CrosstalkResult, compute_crosstalk_pairs

__all__ = ["CrosstalkResult", "compute_crosstalk_pairs"]
