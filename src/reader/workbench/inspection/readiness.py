from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from reader.errors import RecordError
from reader.runtime import ReaderRuntime
from reader.workbench.commands import reader_command
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.engine.planning import build_next_steps
from reader.workbench.engine.validation import validation_summary
from reader.workbench.graph import resolve_workbench

from .common import summarize_outputs_dir

READINESS_STATES = frozenset(
    {
        "config_error",
        "draft",
        "template",
        "dependency_blocked",
        "blocked",
        "runnable",
        "uncataloged_outputs_present",
        "records_ready",
    }
)
READINESS_CAPABILITY_KEYS = frozenset(
    {
        "run",
        "records",
        "plot",
        "export",
        "notebook_scaffold",
        "notebook_scan_records",
    }
)


def config_error_readiness_payload(error: str) -> dict[str, object]:
    return {
        "state": "config_error",
        "summary": "config error",
        "preflight": {
            "status": "error",
            "issues": 1,
            "files": "not checked",
            "dependencies": "not checked",
        },
        "records": {
            "catalog": False,
            "available": False,
            "uncataloged_outputs_present": False,
        },
        "capabilities": {
            "run": False,
            "records": False,
            "plot": False,
            "export": False,
            "notebook_scaffold": False,
            "notebook_scan_records": False,
        },
        "errors": [error],
        "next_steps": [],
    }


def _non_active_lifecycle_payload(*, lifecycle: str, job_path: Path) -> dict[str, object]:
    next_step = {
        "command": reader_command("validate", job_path, "--no-files", "--format", "json"),
        "description": "Check the config shape without treating this non-active experiment as ready to run.",
    }
    summary_by_lifecycle = {
        "draft": "draft experiment; add inputs and switch lifecycle to active when ready",
        "template": "template scaffold; copy/customize before treating it as a runnable experiment",
    }
    return {
        "state": lifecycle,
        "summary": summary_by_lifecycle.get(lifecycle, lifecycle),
        "preflight": {
            "status": "skipped",
            "issues": 0,
            "files": "not evaluated for non-active lifecycle",
            "dependencies": "not evaluated for non-active lifecycle",
        },
        "records": {
            "catalog": False,
            "available": False,
            "uncataloged_outputs_present": False,
        },
        "capabilities": {
            "run": False,
            "records": False,
            "plot": False,
            "export": False,
            "notebook_scaffold": True,
            "notebook_scan_records": False,
        },
        "errors": [],
        "next_steps": [next_step],
    }


def readiness_summary_text(payload: dict[str, object] | None) -> str:
    if not isinstance(payload, dict):
        return "—"
    summary = str(payload.get("summary") or "").strip()
    return summary or str(payload.get("state") or "—")


def readiness_primary_command(payload: dict[str, object] | None) -> str:
    if not isinstance(payload, dict):
        return "—"
    next_steps = payload.get("next_steps") or []
    if not isinstance(next_steps, list) or not next_steps:
        return "—"
    first = next_steps[0]
    if not isinstance(first, dict):
        return "—"
    command = str(first.get("command") or "").strip()
    return command or "—"


def experiment_readiness_payload(
    *,
    job_path: Path,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
    check_files: bool = True,
) -> dict[str, object]:
    if decl.experiment.lifecycle != "active":
        return _non_active_lifecycle_payload(lifecycle=decl.experiment.lifecycle, job_path=job_path)

    summary = validation_summary(
        decl,
        check_files=check_files,
        exp_root=decl.experiment.root,
        runtime=runtime,
    )
    layout = decl.experiment_semantics.layout
    outputs_dir = layout.outputs_dir
    store = runtime.record_store(
        outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        create=False,
    )
    records_catalog = store.catalog_exists()
    records_available = False
    record_catalog_error: str | None = None
    if records_catalog:
        try:
            records_available = bool(store.iter_latest_records())
        except RecordError as exc:
            record_catalog_error = str(exc)
    generated = summarize_outputs_dir(
        outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        notebooks_subdir=layout.notebooks_subdir,
    )
    uncataloged_outputs_present = (
        any(generated[key] for key in ("records", "plots", "exports")) and not records_available
    )
    workbench = resolve_workbench(decl)
    can_run = summary["status"] == "ok"
    file_issues = int(summary["files"].get("issues") or 0)
    dependency_issues = int(summary["dependencies"].get("issues") or 0)
    if not can_run and dependency_issues and not file_issues:
        state = "dependency_blocked"
        summary_text = f"dependency blocked ({dependency_issues} issue(s))"
        next_steps = [
            {
                "command": reader_command("validate", job_path, "--format", "json"),
                "description": "Check missing runtime dependencies before running this assay.",
            }
        ]
    elif not can_run:
        state = "blocked"
        summary_text = f"blocked ({len(summary['errors'])} issue(s))"
        next_steps = [
            {
                "command": reader_command("validate", job_path, "--format", "json"),
                "description": "Check blocking file or dependency issues.",
            }
        ]
    elif record_catalog_error is not None:
        state = "blocked"
        summary_text = "blocked (invalid records catalog)"
        next_steps = [
            {
                "command": reader_command("records", job_path, "--format", "json"),
                "description": "Inspect and repair the invalid records catalog before using persisted outputs.",
            }
        ]
    elif records_available:
        state = "records_ready"
        summary_text = "records ready"
        next_steps = [
            {"command": command, "description": description}
            for command, description in build_next_steps(decl, job_label=str(job_path), runtime=runtime)
        ]
    elif uncataloged_outputs_present:
        state = "uncataloged_outputs_present"
        summary_text = "generated outputs present without usable current records"
        next_steps = [
            {
                "command": reader_command("run", job_path),
                "description": "Rerun from source inputs to rebuild records and selected outputs.",
            }
        ]
    else:
        state = "runnable"
        summary_text = "ready to run"
        next_steps = [
            {
                "command": reader_command("run", job_path),
                "description": "Generate dataframe records and selected outputs.",
            }
        ]
    errors = deepcopy(summary["errors"])
    if record_catalog_error is not None:
        errors.append(f"records catalog: {record_catalog_error}")
    return {
        "state": state,
        "summary": summary_text,
        "preflight": {
            "status": summary["status"],
            "issues": len(errors),
            "files": summary["files"]["summary"],
            "dependencies": summary["dependencies"]["summary"],
        },
        "records": {
            "catalog": records_catalog,
            "available": records_available,
            "uncataloged_outputs_present": uncataloged_outputs_present,
        },
        "capabilities": {
            "run": can_run,
            "records": records_available,
            "plot": records_available and bool(workbench.plots),
            "export": records_available and bool(workbench.exports),
            "notebook_scaffold": bool(workbench.notebooks),
            "notebook_scan_records": records_available and bool(workbench.notebooks),
        },
        "errors": errors,
        "next_steps": next_steps,
    }
