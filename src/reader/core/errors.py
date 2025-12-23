"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/errors.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class ReaderError(Exception): ...


class ConfigError(ReaderError): ...


class RegistryError(ReaderError): ...


class ContractError(ReaderError): ...


class ArtifactError(ReaderError): ...


class ExecutionError(ReaderError): ...


class ParseError(ReaderError): ...


class MergeError(ReaderError): ...


class TransformError(ReaderError): ...


class PlotError(ReaderError): ...


class SFXIError(ReaderError): ...
