"""Package-wide exception taxonomy."""

from __future__ import annotations


class ReaderError(Exception): ...


class ConfigError(ReaderError): ...


class RegistryError(ReaderError): ...


class ContractError(ReaderError): ...


class RecordError(ReaderError): ...


class ProvenanceEpochChangedError(RecordError):
    """A bound record-store operation observed a different provenance epoch."""


class ExecutionError(ReaderError): ...


class InvocationFinalizationError(ExecutionError):
    """Execution committed records but could not confirm its terminal ledger event."""

    def __init__(
        self,
        message: str,
        *,
        invocation_id: str,
        produced_record_revisions: tuple[dict[str, object], ...],
    ) -> None:
        super().__init__(message)
        self.invocation_id = invocation_id
        self.produced_record_revisions = produced_record_revisions


class ParseError(ReaderError): ...


class MergeError(ReaderError): ...


class TransformError(ReaderError): ...


class PlotError(ReaderError): ...


class FourStateVectorError(ReaderError): ...
