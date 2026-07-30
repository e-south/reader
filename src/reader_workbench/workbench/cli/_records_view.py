from __future__ import annotations

from rich import box
from rich.panel import Panel

from reader_workbench.errors import ReaderError
from reader_workbench.workbench.commands import reader_command

from . import shared
from ._lazy import load as _load
from .helpers import load_job_models
from .shared import emit_json, normalize_output_format, table


def render_records(
    *,
    job_path,
    all_revisions: bool,
    format: str,
    limit: int | None = None,
    continuation: str | None = None,
) -> None:
    _, decl = load_job_models(job_path)
    outputs_dir = decl.experiment_semantics.layout.outputs_dir
    runtime = _load("reader_workbench.runtime").builtin_runtime()
    store = runtime.record_store(
        outputs_dir,
        plots_subdir=decl.experiment_semantics.layout.plots_subdir,
        exports_subdir=decl.experiment_semantics.layout.exports_subdir,
        create=False,
    )
    if not store.catalog_exists():
        raise ReaderError(
            f"No outputs/manifests/records.json found. Run '{reader_command('run', job_path)}' first to produce records."
        )

    fmt = normalize_output_format(format)
    limit, continuation = shared.normalize_paging_options(limit, continuation)
    shared.require_json_paging(format=fmt, limit=limit, continuation=continuation)
    if fmt == "json":
        payload = _load("reader_workbench.workbench.inspection.results").record_catalog_payload(
            experiment=_load("reader_workbench.workbench.inspection.experiments").experiment_identity_payload(
                job_path=job_path, decl=decl
            ),
            store=store,
            outputs_dir=outputs_dir,
            runtime=runtime,
            include_history=all_revisions,
        )
        page = shared.page_json_collection(
            payload["records"],
            key=lambda item: str(item["record_id"]),
            surface="records",
            selection={
                "config": str(job_path),
                "include_history": all_revisions,
            },
            limit=limit,
            continuation=continuation,
        )
        payload["records"] = list(page.items)
        emit_json(payload, truncated=page.truncated, continuation=page.continuation)
        return

    latest_records = store.iter_latest_records()
    if all_revisions:
        if not latest_records:
            shared.console.print(
                Panel.fit(
                    (
                        "No record history listed in outputs/manifests/records.json. "
                        f"Run '{reader_command('run', job_path)}' first."
                    ),
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        revision_counts = store.revision_counts(record.record_id for record in latest_records)
        listing = table("Records • history")
        listing.add_column("Record")
        listing.add_column("Kind", style="accent")
        listing.add_column("Producer")
        listing.add_column("Description")
        listing.add_column("Revisions", justify="right")
        for record in latest_records:
            listing.add_row(
                record.record_id,
                record.kind,
                f"{record.producer.kind}:{record.producer.id}",
                _load("reader_workbench.workbench.inspection.results").record_description(record, runtime=runtime),
                str(revision_counts[record.record_id]),
            )
        shared.console.print(Panel(listing, border_style="accent", box=box.ROUNDED))
        return

    if not latest_records:
        shared.console.print(
            Panel.fit(
                (
                    "No records listed in outputs/manifests/records.json. "
                    f"Run '{reader_command('run', job_path)}' first."
                ),
                border_style="warn",
                box=box.ROUNDED,
            )
        )
        return
    listing = table("Records • latest")
    listing.add_column("Record")
    listing.add_column("Kind", style="accent")
    listing.add_column("Producer")
    listing.add_column("Description")
    listing.add_column("Details", style="path")
    for record in latest_records:
        detail = _load("reader_workbench.workbench.inspection.results").record_detail_text(
            record,
            base=decl.experiment.root,
        )
        listing.add_row(
            record.record_id,
            record.kind,
            f"{record.producer.kind}:{record.producer.id}",
            _load("reader_workbench.workbench.inspection.results").record_description(record, runtime=runtime),
            detail,
        )
    shared.console.print(Panel(listing, border_style="accent", box=box.ROUNDED))
