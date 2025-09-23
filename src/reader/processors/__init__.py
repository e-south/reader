"""
--------------------------------------------------------------------------------
<reader project>
reader/processors/__init__.py

Unified transform registry & sequential executor.

Usage:
    • Register transforms with @register_transform("name")
    • Call apply_transform_sequence(df, transforms_cfg, reader_cfg)
      → returns (df_final, TransformContext)

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import pandas as pd

from reader.config import ReaderCfg, XForm

LOG = logging.getLogger(__name__)

# ── registry ───────────────────────────────────────────────────────────────

TransformFn = Callable[[pd.DataFrame, XForm, "TransformContext", ReaderCfg], pd.DataFrame]
_TRANSFORM_REGISTRY: Dict[str, TransformFn] = {}


def register_transform(name: str) -> Callable[[TransformFn], TransformFn]:
    def _wrap(fn: TransformFn) -> TransformFn:
        _TRANSFORM_REGISTRY[name] = fn
        return fn
    return _wrap


def get_transform(name: str):
    # Allow configs to include `- type: sfxi` without requiring a transform.
    # The actual SFXI ingestion runs later via reader.processors.sfxi.run_sfxi().
    if name == "sfxi":
        def _noop(*args, **kwargs):
            LOG.info("sfxi: noop transform (real SFXI work happens in post-step)")
            # return original df as-is; keep contract identical to other transforms
            if args:
                return args[0]
            return kwargs.get("df")
        return _noop
    try:
        return _TRANSFORM_REGISTRY[name]
    except KeyError:
        raise ValueError(f"No transform registered for type '{name}'")


# ── context ────────────────────────────────────────────────────────────────

@dataclass
class TransformContext:
    """Carrier for side artifacts produced by transforms."""
    blanks: pd.DataFrame | None = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def set_default_blanks(self, df_like: pd.DataFrame):
        if self.blanks is None:
            self.blanks = df_like.iloc[0:0].copy()


# ── import all transforms once (for decorator side-effects) ────────────────

def ensure_all_transforms_imported() -> None:
    """
    Import submodules inside `reader.processors` so that @register_transform
    decorators run. Skip 'sfxi' package and private modules.
    """
    pkg = Path(__file__).parent
    for _, module_name, ispkg in pkgutil.iter_modules([str(pkg)]):
        if module_name.startswith("_"):
            continue
        if module_name in {"__init__", "merger", "logic_symmetry_prep"}:
            continue
        if module_name == "sfxi" or ispkg:
            continue
        importlib.import_module(f"reader.processors.{module_name}")


# ── executor ───────────────────────────────────────────────────────────────

def apply_transform_sequence(
    df: pd.DataFrame,
    transforms_cfg: Iterable[XForm],
    reader_cfg: ReaderCfg,
) -> Tuple[pd.DataFrame, TransformContext]:
    """
    Apply transforms in the order provided by YAML `transformations:`.

    Each transform function:
        fn(df, cfg_obj, ctx, reader_cfg) -> df_new
    """
    ctx = TransformContext()
    current = df.copy()
    for xf in transforms_cfg:
        fn = get_transform(xf.type)
        current = fn(current, xf, ctx, reader_cfg)
    return current, ctx
