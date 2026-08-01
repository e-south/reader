from __future__ import annotations

import io
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console

from reader_workbench.errors import ReaderError
from reader_workbench.runtime import builtin_runtime
from reader_workbench.workbench.decl import load_workbench_decl
from reader_workbench.workbench.engine import run_spec
from reader_workbench.workbench.engine.validation import validation_summary
from reader_workbench.workbench.experiments import discover_experiment_configs
from reader_workbench.workbench.graph import resolve_workbench
from reader_workbench.workbench.inspection.runtime import record_producer_map, workbench_record_verification_scope
from reader_workbench.workbench.records import verify_record_store

from .staging import stage_audit_workspace


@dataclass(slots=True)
class AuditResult:
    config: str
    name: str
    lifecycle: str
    status: str
    phase: str
    seconds: float
    detail: str | None
    expected_plots: int
    expected_exports: int


def discover_year_dirs(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit() and len(path.name) == 4)


def discover_configs(root: Path, years: set[str]) -> list[Path]:
    configs = discover_experiment_configs(root, include_scaffolds=False)
    selected: list[Path] = []
    for config in configs:
        try:
            rel = config.relative_to(root)
        except ValueError:
            continue
        if not rel.parts:
            continue
        if years and rel.parts[0] not in years:
            continue
        selected.append(config)
    return sorted(selected)


def verify_outputs(decl, runtime) -> tuple[int, int, str | None]:
    workbench = resolve_workbench(decl)
    verification_scope = workbench_record_verification_scope(workbench, runtime=runtime)
    layout = decl.experiment_semantics.layout
    outputs = layout.outputs_dir
    store = runtime.record_store(
        outputs,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=decl.experiment.root,
        create=False,
    )
    latest_ids = {record.record_id for record in store.iter_latest_records()}
    expected_plot_ids = {f"plot:{plot.id}" for plot in workbench.plots}
    expected_export_ids = {f"export:{export.id}" for export in workbench.exports}
    expected_dataframe_ids = set(record_producer_map(workbench.plugin_steps(), runtime=runtime))

    if (outputs / "manifests" / "records.json").exists() is False:
        return len(expected_plot_ids), len(expected_export_ids), "records.json missing"
    missing_dataframe_ids = sorted(expected_dataframe_ids - latest_ids)
    if missing_dataframe_ids:
        return (
            len(expected_plot_ids),
            len(expected_export_ids),
            f"missing declared dataframe records: {missing_dataframe_ids}",
        )

    missing_plot_ids = sorted(expected_plot_ids - latest_ids)
    if missing_plot_ids:
        return len(expected_plot_ids), len(expected_export_ids), f"missing plot records: {missing_plot_ids}"
    missing_export_ids = sorted(expected_export_ids - latest_ids)
    if missing_export_ids:
        return len(expected_plot_ids), len(expected_export_ids), f"missing export records: {missing_export_ids}"

    plots_dir = outputs / layout.plots_subdir
    exports_dir = outputs / layout.exports_subdir
    if expected_plot_ids and not any(path.is_file() for path in plots_dir.rglob("*")):
        return len(expected_plot_ids), len(expected_export_ids), "expected plot files were not created"
    if expected_export_ids and not any(path.is_file() for path in exports_dir.rglob("*")):
        return len(expected_plot_ids), len(expected_export_ids), "expected export files were not created"

    verification = verify_record_store(
        store,
        experiment_root=decl.experiment.root,
        expected_config_digest=decl.config_digest,
        scope=verification_scope,
    )
    if verification["status"] != "ok":
        issues = list(verification.get("issues") or [])
        for record in verification.get("records") or []:
            issues.extend(record.get("issues") or [])
        first = issues[0] if issues else {"code": "verification.failed", "reason": "unknown verification failure"}
        return (
            len(expected_plot_ids),
            len(expected_export_ids),
            f"reader verify failed: {first['code']} • {first['reason']}",
        )

    return len(expected_plot_ids), len(expected_export_ids), None


def audit_config(config_path: Path, *, include_non_active: bool, runtime) -> AuditResult:
    start = time.perf_counter()
    try:
        decl = load_workbench_decl(config_path, protocols=runtime.protocols)
    except ReaderError as exc:
        return AuditResult(
            config=str(config_path),
            name=config_path.parent.name,
            lifecycle="unknown",
            status="failed",
            phase="config",
            seconds=time.perf_counter() - start,
            detail=str(exc),
            expected_plots=0,
            expected_exports=0,
        )
    lifecycle = decl.experiment.lifecycle
    rel_config = str(config_path)
    if lifecycle != "active" and not include_non_active:
        return AuditResult(
            config=rel_config,
            name=config_path.parent.name,
            lifecycle=lifecycle,
            status="skipped",
            phase="lifecycle",
            seconds=time.perf_counter() - start,
            detail=f"skipped non-active lifecycle: {lifecycle}",
            expected_plots=0,
            expected_exports=0,
        )

    source_summary = validation_summary(
        decl,
        check_files=True,
        exp_root=decl.experiment.root,
        runtime=runtime,
    )
    if source_summary["status"] != "ok":
        return AuditResult(
            config=rel_config,
            name=config_path.parent.name,
            lifecycle=lifecycle,
            status="failed",
            phase="validate",
            seconds=time.perf_counter() - start,
            detail="; ".join(source_summary["errors"]) or "validation failed",
            expected_plots=0,
            expected_exports=0,
        )

    with tempfile.TemporaryDirectory(prefix="reader-local-audit-") as tmpdir:
        try:
            staged_config = stage_audit_workspace(
                config_path=config_path,
                target_root=Path(tmpdir),
                resources=decl.experiment_semantics.resources,
                runtime=runtime,
            )
            staged_decl = load_workbench_decl(staged_config, protocols=runtime.protocols)
        except (ReaderError, OSError) as exc:
            return AuditResult(
                config=rel_config,
                name=config_path.parent.name,
                lifecycle=lifecycle,
                status="failed",
                phase="validate",
                seconds=time.perf_counter() - start,
                detail=str(exc),
                expected_plots=0,
                expected_exports=0,
            )
        summary = validation_summary(
            staged_decl, check_files=True, exp_root=staged_decl.experiment.root, runtime=runtime
        )
        if summary["status"] != "ok":
            return AuditResult(
                config=rel_config,
                name=config_path.parent.name,
                lifecycle=lifecycle,
                status="failed",
                phase="validate",
                seconds=time.perf_counter() - start,
                detail="; ".join(summary["errors"]) or "validation failed",
                expected_plots=0,
                expected_exports=0,
            )

        try:
            run_spec(
                staged_decl,
                include_pipeline=True,
                include_plots=True,
                include_exports=True,
                runtime=runtime,
                console=Console(file=io.StringIO(), force_terminal=False, color_system=None),
                log_level="ERROR",
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - exercised by live audit runs
            return AuditResult(
                config=rel_config,
                name=config_path.parent.name,
                lifecycle=lifecycle,
                status="failed",
                phase="run",
                seconds=time.perf_counter() - start,
                detail=str(exc),
                expected_plots=0,
                expected_exports=0,
            )

        expected_plots, expected_exports, verify_error = verify_outputs(staged_decl, runtime)
        if verify_error is not None:
            return AuditResult(
                config=rel_config,
                name=config_path.parent.name,
                lifecycle=lifecycle,
                status="failed",
                phase="verify",
                seconds=time.perf_counter() - start,
                detail=verify_error,
                expected_plots=expected_plots,
                expected_exports=expected_exports,
            )

        return AuditResult(
            config=rel_config,
            name=config_path.parent.name,
            lifecycle=lifecycle,
            status="passed",
            phase="complete",
            seconds=time.perf_counter() - start,
            detail=None,
            expected_plots=expected_plots,
            expected_exports=expected_exports,
        )


def render_text(results: list[AuditResult]) -> str:
    lines = []
    for result in results:
        detail = f" :: {result.detail}" if result.detail else ""
        lines.append(
            f"[{result.status.upper():7}] {result.name} "
            f"(lifecycle={result.lifecycle}, phase={result.phase}, "
            f"plots={result.expected_plots}, exports={result.expected_exports}, "
            f"seconds={result.seconds:.2f}){detail}"
        )
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    lines.append(f"Summary: {summary}")
    return "\n".join(lines)


def audit_experiments(
    root: Path,
    *,
    years: tuple[str, ...] | None = None,
    include_non_active: bool = False,
    fail_fast: bool = False,
) -> dict[str, object]:
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Experiments root not found: {root}")
    selected_years = discover_year_dirs(root) if years is None else [str(year) for year in years]
    if not selected_years:
        raise ValueError(f"No numeric year directories found under {root}. Use --years to select specific directories.")
    runtime = builtin_runtime()
    configs = discover_configs(root, set(selected_years))

    results: list[AuditResult] = []
    for config_path in configs:
        result = audit_config(config_path, include_non_active=include_non_active, runtime=runtime)
        results.append(result)
        if fail_fast and result.status == "failed":
            break

    return {
        "root": str(root),
        "years": sorted(selected_years),
        "summary": {
            "experiments": len(results),
            "passed": sum(1 for item in results if item.status == "passed"),
            "failed": sum(1 for item in results if item.status == "failed"),
            "skipped": sum(1 for item in results if item.status == "skipped"),
        },
        "results": [asdict(result) for result in results],
    }
