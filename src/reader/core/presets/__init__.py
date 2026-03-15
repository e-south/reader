"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/presets/__init__.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .registry import PRESETS, describe_preset, infer_category, list_presets, resolve_preset

__all__ = ["PRESETS", "describe_preset", "infer_category", "list_presets", "resolve_preset"]
