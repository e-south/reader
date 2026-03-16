"""
--------------------------------------------------------------------------------
<reader project>
src/reader/errors.py

Package-wide exception taxonomy.
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class ReaderError(Exception): ...


class ConfigError(ReaderError): ...


class RegistryError(ReaderError): ...


class ContractError(ReaderError): ...


class RecordError(ReaderError): ...


class ExecutionError(ReaderError): ...


class ParseError(ReaderError): ...


class MergeError(ReaderError): ...


class TransformError(ReaderError): ...


class PlotError(ReaderError): ...


class SFXIError(ReaderError): ...
