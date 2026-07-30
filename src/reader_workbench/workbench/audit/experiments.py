from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console

from reader_workbench.runtime import builtin_runtime
from reader_workbench.workbench.decl import load_workbench_decl
from reader_workbench.workbench.engine import run_spec
from reader_workbench.workbench.engine.validation import validation_summary
from reader_workbench.workbench.experiments import discover_experiment_configs
from reader_workbench.workbench.graph import resolve_workbench
from reader_workbench.workbench.inspection.runtime import workbench_record_verification_scope
from reader_workbench.workbench.records import verify_record_store


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


def stage_experiment(source_dir: Path, target_dir: Path) -> Path:
    shutil.copytree(
        source_dir,
        target_dir,
        ignore=shutil.ignore_patterns("outputs", "__pycache__", ".DS_Store"),
        symlinks=True,
    )
    _retarget_internal_symlinks(source_dir=source_dir, target_dir=target_dir)
    return target_dir / "config.yaml"


def _retarget_internal_symlinks(*, source_dir: Path, target_dir: Path) -> None:
    """Map confined source symlinks into the staged tree without following links while copying."""

    source_root = source_dir.resolve(strict=True)
    target_root = target_dir.resolve(strict=True)
    for current, dirnames, filenames in os.walk(target_root, followlinks=False):
        current_path = Path(current)
        for name in [*dirnames, *filenames]:
            staged_link = current_path / name
            if not staged_link.is_symlink():
                continue
            relative_link = staged_link.relative_to(target_root)
            source_link = source_root / relative_link
            try:
                source_target = source_link.resolve(strict=True)
                relative_target = source_target.relative_to(source_root)
            except (OSError, RuntimeError, ValueError):
                # External, dangling, or cyclic links remain links. Staged
                # validation can then reject them without copytree traversing them.
                continue
            staged_target = target_root / relative_target
            link_target = os.path.relpath(staged_target, start=staged_link.parent)
            target_is_directory = source_target.is_dir()
            staged_link.unlink()
            staged_link.symlink_to(link_target, target_is_directory=target_is_directory)


def verify_outputs(decl, runtime) -> tuple[int, int, str | None]:
    workbench = resolve_workbench(decl)
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

    if (outputs / "manifests" / "records.json").exists() is False:
        return len(expected_plot_ids), len(expected_export_ids), "records.json missing"
    if "ingest/df" not in latest_ids:
        return len(expected_plot_ids), len(expected_export_ids), "ingest/df record missing"

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
        scope=workbench_record_verification_scope(workbench, runtime=runtime),
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
    decl = load_workbench_decl(config_path, protocols=runtime.protocols)
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
        staged_config = stage_experiment(config_path.parent, Path(tmpdir) / config_path.parent.name)
        staged_decl = load_workbench_decl(staged_config, protocols=runtime.protocols)
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
