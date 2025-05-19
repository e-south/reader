"""
--------------------------------------------------------------------------------
<reader project>
reader/parsers/raw.py

BaseRawParser and registry for raw-data parsers.

- `BaseRawParser`: abstract interface that all raw parsers must implement.
- `register_raw_parser(name)`: decorator to register a parser.
- `get_raw_parser(name)`: lookup by key.
- `ensure_all_parsers_imported()`: dynamically import all parser modules so decorators run.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Optional, List, Type, Dict, Callable

import pkgutil
import importlib
import pandas as pd

from reader.processors.merger import merge_raw_and_map

logger = logging.getLogger(__name__)

class BaseRawParser(ABC):
    """Abstract interface for raw-data parsers, now with a merge hook."""

    def __init__(
        self,
        path: Path | str,
        channel_map: Mapping[str, str] | None = None,
        channels: Optional[List[str]] = None,
    ):
        self.path = Path(path)
        self.channel_map = channel_map or {}
        self.channels = channels

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """
        Read `self.path` and return a tidy DataFrame with at least
        a 'position' column.
        """
        ...

    def merge(self, raw_df: pd.DataFrame, plate_map_df: pd.DataFrame) -> pd.DataFrame:
        """
        Default merge‑hook: delegate to the generic merger.
        Parsers needing custom logic can override this.
        """
        logger.debug("Using default merge hook for parser %s", self.__class__.__name__)
        return merge_raw_and_map(raw_df, plate_map_df)


# Registry for raw parsers
RAW_PARSER_REGISTRY: Dict[str, Type[BaseRawParser]] = {}

def register_raw_parser(name: str) -> Callable[[Type[BaseRawParser]], Type[BaseRawParser]]:
    """Decorator to register a parser class by key."""
    def _wrap(cls: Type[BaseRawParser]) -> Type[BaseRawParser]:
        RAW_PARSER_REGISTRY[name] = cls
        logger.debug("Registered raw parser '%s' → %s", name, cls.__name__)
        return cls
    return _wrap

def get_raw_parser(name: str) -> Type[BaseRawParser]:
    """Lookup a registered parser by name."""
    try:
        return RAW_PARSER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown raw parser '{name}'. Available: {list(RAW_PARSER_REGISTRY)}")

def ensure_all_parsers_imported() -> None:
    """Import all parser modules so that registration decorators run."""
    pkg = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(pkg)]):
        if module_name != "raw":
            importlib.import_module(f"reader.parsers.{module_name}")
            logger.debug("Imported parser module reader.parsers.%s", module_name)
