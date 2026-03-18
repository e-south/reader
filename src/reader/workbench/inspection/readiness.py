from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from reader.runtime import ReaderRuntime
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.engine.planning import build_next_steps
from reader.workbench.engine.validation import validation_summary
from reader.workbench.graph import resolve_workbench


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
    workbench = resolve_workbench(decl)
    can_run = summary["status"] == "ok"
    if not can_run:
        state = "blocked"
        summary_text = f"blocked ({len(summary['errors'])} issue(s))"
        next_steps = [
            {
                "command": f"reader validate {job_path} --format json",
                "description": "Inspect blocking file or dependency issues.",
            }
        ]
    elif records_catalog:
        state = "records_ready"
        summary_text = "records ready"
        next_steps = [
            {"command": command, "description": description}
            for command, description in build_next_steps(decl, job_label=str(job_path), runtime=runtime)
        ]
    else:
        state = "runnable"
        summary_text = "ready to run"
        next_steps = [
            {
                "command": f"reader run {job_path}",
                "description": "Generate dataframe records and selected outputs.",
            }
        ]
    return {
        "state": state,
        "summary": summary_text,
        "preflight": {
            "status": summary["status"],
            "issues": len(summary["errors"]),
            "files": summary["files"]["summary"],
            "dependencies": summary["dependencies"]["summary"],
        },
        "records": {
            "catalog": records_catalog,
        },
        "capabilities": {
            "run": can_run,
            "records": records_catalog,
            "plot": records_catalog and bool(workbench.plots),
            "export": records_catalog and bool(workbench.exports),
            "notebook_scaffold": bool(workbench.notebooks),
            "notebook_scan_records": records_catalog and bool(workbench.notebooks),
        },
        "errors": deepcopy(summary["errors"]),
        "next_steps": next_steps,
    }
