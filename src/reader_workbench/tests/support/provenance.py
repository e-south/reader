from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reader_workbench.workbench.engine.invocations import InvocationLedger
from reader_workbench.workbench.records import RecordStore, WorkbenchRecord, record_revision_digest
from reader_workbench.workbench.records.identity import current_build_identity


def record_successful_invocation(
    store: RecordStore,
    *,
    records: Sequence[WorkbenchRecord],
    config_digest: str,
    operation: str,
    selected_step_ids: dict[str, list[str]],
    declared_inputs: list[dict[str, Any]] | None = None,
) -> None:
    """Claim freshly persisted test records through the production ledger contract."""

    ledger = InvocationLedger.for_store(store=store)
    attempt = ledger.append_attempt(
        config_digest=config_digest,
        build_identity=current_build_identity(),
        operation=operation,
        selected_step_ids=selected_step_ids,
        declared_inputs=[] if declared_inputs is None else declared_inputs,
    )
    ledger.append_result(
        attempt,
        exit_status=0,
        produced_record_revisions=[
            {
                "record_id": record.record_id,
                "revision": len(store.record_history(record.record_id)),
                "revision_digest": record_revision_digest(record, outputs_dir=store.root),
            }
            for record in records
        ],
    )
