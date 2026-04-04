from __future__ import annotations

from functools import cache
from importlib import import_module


@cache
def load(module_name: str):
    return import_module(module_name)
