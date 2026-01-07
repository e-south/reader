"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/sfxi/__init__.py

SFXI internals used by the transform plugin.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .math import compute_vec8
from .reference import resolve_reference_design_id
from .selection import cornerize_and_aggregate

__all__ = ["compute_vec8", "cornerize_and_aggregate", "resolve_reference_design_id"]
