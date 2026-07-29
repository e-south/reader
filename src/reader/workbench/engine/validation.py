from __future__ import annotations

from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel

from reader.errors import ConfigError, ExecutionError, ReaderError
from reader.plotting.mpl import ensure_mpl_cache_dir
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.commands import reader_command
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.experiment import ExperimentSemantics
from reader.workbench.graph import (
    FileRef,
    RecordCollectionRef,
    RecordRef,
    ResourceRef,
    ensure_unique_workbench_ids,
    input_ref_display,
    resolve_workbench,
)
from reader.workbench.paths import resolve_path_within_root
from reader.workbench.records import DataFrameArtifactRecord, resolve_source_record
from reader.workbench.registry import Plugin, PreflightIssue
from reader.workbench.templates import require_notebook_template_for_protocol

from ._shared import collect_categories
from .contracts import _resolve_output_labels


def _validate_reads_shape(*, label: str, step: Any, plugin_cls: Any) -> None:
    try:
        required = plugin_cls.input_ports()
        expected = set(required)
        mandatory = {name for name, port in required.items() if not port.optional}
        provided = set((step.reads or {}).keys())
        missing = sorted(mandatory - provided)
        extra = sorted(provided - expected)
        if not missing and not extra:
            return
        parts = []
        if missing:
            parts.append(f"missing inputs: {missing}")
        if extra:
            parts.append(f"unexpected inputs: {extra}")
        raise ConfigError(f"{label} {step.id}: reads do not match plugin inputs ({'; '.join(parts)})")
    except ConfigError:
        raise
    except Exception as err:
        raise ConfigError(f"{label} {step.id}: could not validate reads: {err}") from err


def _assert_known_read_targets(*, label: str, step: Any, available_labels: set[str], plugin_cls: Any) -> None:
    input_ports = plugin_cls.input_ports()
    for key, target in (step.reads or {}).items():
        port = input_ports.get(key)
        if port is None:
            continue
        if port.kind in {"dataframe", "file_bundle"} and isinstance(target, FileRef | ResourceRef):
            raise ConfigError(
                f"{label} {step.id}: reads '{key}' targets '{input_ref_display(target)}', "
                f"but the plugin expects a {port.kind} record."
            )
        if port.kind == "record_collection" and not isinstance(target, RecordCollectionRef):
            raise ConfigError(f"{label} {step.id}: reads '{key}' must target a declared record collection.")
        if isinstance(target, RecordCollectionRef):
            if port.kind != "record_collection":
                raise ConfigError(
                    f"{label} {step.id}: reads '{key}' targets a record collection, "
                    f"but the plugin expects {port.kind!r}."
                )
            continue
        if port.kind in {"file_path", "file_set"} and isinstance(target, RecordRef):
            raise ConfigError(
                f"{label} {step.id}: reads '{key}' targets record '{target.record_id}', "
                "but the plugin expects an explicit file/resource binding."
            )
        if isinstance(target, FileRef | ResourceRef):
            continue
        if isinstance(target, RecordRef) and target.record_id in available_labels:
            continue
        preview = sorted(available_labels)
        shown = ", ".join(preview[:12]) if preview else "—"
        tail = " …" if len(preview) > 12 else ""
        source = "pipeline" if label in {"plot", "export"} else "any prior step"
        raise ConfigError(
            f"{label} {step.id}: reads '{key}' → '{input_ref_display(target)}', which is not produced by {source}. "
            f"Known labels{' so far' if label == 'pipeline' else ''}: {shown}{tail}. "
            "Check writes/reads aliases or use explicit file/resource bindings."
        )


def _effective_step_with(*, protocol: Any, step: Any) -> dict[str, Any]:
    return protocol.effective_plugin_config(plugin_id=step.plugin, step_with=(step.with_ or {}))


def _validate_step_config(*, label: str, step: Any, plugin_cls: Any, protocol: Any) -> dict[str, Any]:
    with_block = _effective_step_with(protocol=protocol, step=step)
    try:
        plugin_cls.ConfigModel.model_validate(with_block)
    except Exception as err:
        raise ConfigError(f"{label} {step.id}: invalid config for {step.plugin}: {err}") from err
    return with_block


def _validate_plot_semantic_refs(*, step: Any, experiment: ExperimentSemantics, with_block: dict[str, Any]) -> None:
    partition = with_block.get("partition")
    if partition is not None:
        try:
            experiment.annotations.resolve_plot_partition(partition=partition)
        except Exception as err:
            raise ConfigError(f"plot {step.id}: invalid plot partition: {err}") from err

    x_column = with_block.get("x")
    y_column = with_block.get("y")
    try:
        experiment.annotations.resolve_order_arg(
            order=with_block.get("order_x"),
            order_ref=with_block.get("order_x_ref"),
            column=(x_column if isinstance(x_column, str) else None),
            arg_name="order_x",
        )
        experiment.annotations.resolve_order_arg(
            order=with_block.get("order_y"),
            order_ref=with_block.get("order_y_ref"),
            column=(y_column if isinstance(y_column, str) else None),
            arg_name="order_y",
        )
        if isinstance(with_block.get("state_map_ref"), str):
            experiment.annotations.resolve_ordered_state_space(ref=with_block["state_map_ref"])
    except Exception as err:
        raise ConfigError(f"plot {step.id}: invalid ordering semantic reference: {err}") from err


def _validate_pipeline_semantic_refs(*, step: Any, experiment: ExperimentSemantics, with_block: dict[str, Any]) -> None:
    try:
        if isinstance(with_block.get("state_map_ref"), str):
            experiment.annotations.resolve_ordered_state_space(ref=with_block["state_map_ref"])
    except Exception as err:
        raise ConfigError(f"pipeline {step.id}: invalid annotation semantic reference: {err}") from err


def _validate_pipeline_steps(
    items: list[Any], *, registry: Any, experiment: ExperimentSemantics, protocol: Any
) -> set[str]:
    available: set[str] = set()
    produced: set[str] = set()
    for step in items:
        try:
            plugin_cls = registry.resolve(step.plugin)
        except Exception as err:
            raise ConfigError(f"pipeline {step.id}: {err}") from err
        _validate_reads_shape(label="pipeline", step=step, plugin_cls=plugin_cls)
        _assert_known_read_targets(
            label="pipeline",
            step=step,
            available_labels=available | produced,
            plugin_cls=plugin_cls,
        )
        with_block = _validate_step_config(label="pipeline", step=step, plugin_cls=plugin_cls, protocol=protocol)
        _validate_pipeline_semantic_refs(step=step, experiment=experiment, with_block=with_block)
        try:
            output_labels = _resolve_output_labels(
                step_id=step.id,
                output_ports=plugin_cls.output_ports(),
                writes=(step.writes or {}),
            )
        except ExecutionError as err:
            raise ConfigError(f"pipeline {step.id}: {err}") from err
        for out_name, out_label in output_labels.items():
            if plugin_cls.output_ports()[out_name].kind != "dataframe":
                continue
            if out_label.record_id in available or out_label.record_id in produced:
                raise ConfigError(
                    f"pipeline {step.id}: output label '{out_label.record_id}' is already produced by another step. "
                    "Use a unique writes mapping to avoid clobbering dataframe records."
                )
            produced.add(out_label.record_id)
    return available | produced


def _validate_specs(
    items: list[Any],
    *,
    label: str,
    registry: Any,
    available_labels: set[str],
    protocol: Any,
    experiment: ExperimentSemantics | None = None,
) -> None:
    for spec_item in items:
        try:
            plugin_cls = registry.resolve(spec_item.plugin)
        except Exception as err:
            raise ConfigError(f"{label} {spec_item.id}: {err}") from err
        _validate_reads_shape(label=label, step=spec_item, plugin_cls=plugin_cls)
        _assert_known_read_targets(
            label=label,
            step=spec_item,
            available_labels=available_labels,
            plugin_cls=plugin_cls,
        )
        with_block = _validate_step_config(label=label, step=spec_item, plugin_cls=plugin_cls, protocol=protocol)
        if label == "plot":
            if experiment is None:
                raise ConfigError("plot validation requires experiment semantics")
            _validate_plot_semantic_refs(
                step=spec_item,
                experiment=experiment,
                with_block=with_block,
            )
            try:
                cfg = plugin_cls.ConfigModel.model_validate(with_block)
                plugin_cls.validate_semantic_references(experiment=experiment, cfg=cfg)
            except Exception as err:
                raise ConfigError(f"plot {spec_item.id}: invalid plugin semantic reference: {err}") from err
        data_outputs = {
            key: port.render() for key, port in plugin_cls.output_ports().items() if port.kind == "dataframe"
        }
        if data_outputs:
            raise ConfigError(
                f"{label} {spec_item.id}: plugins must not declare data outputs (got {data_outputs}). "
                "Move data outputs into the pipeline."
            )


def _validate_notebook_specs(items: list[Any], *, protocol: Any) -> None:
    for spec_item in items:
        require_notebook_template_for_protocol(spec_item.template, protocol=protocol)


def _resolve_exp_path(path: Path, *, exp_root: Path | None) -> Path:
    if exp_root is None:
        return path
    try:
        return resolve_path_within_root(path, root=exp_root)
    except ValueError as err:
        raise ConfigError("Declared paths must stay under the experiment root after resolving symlinks.") from err


def _render_rel(path: Path, *, exp_root: Path | None) -> Path:
    if exp_root is None:
        return path
    try:
        return path.relative_to(exp_root)
    except Exception:
        return path


def _plugin_preflight_issues(
    *,
    items: list[Any],
    label: str,
    registry: Any,
    protocol: Any,
    exp_root: Path | None,
) -> list[tuple[str, str, PreflightIssue]]:
    if exp_root is None:
        return []
    issues: list[tuple[str, str, PreflightIssue]] = []
    for step in items:
        plugin_cls = registry.resolve(step.plugin)
        if getattr(plugin_cls.preflight_readiness, "__func__", plugin_cls.preflight_readiness) is getattr(
            Plugin.preflight_readiness,
            "__func__",
            Plugin.preflight_readiness,
        ):
            continue
        cfg = plugin_cls.ConfigModel.model_validate(_effective_step_with(protocol=protocol, step=step))
        for issue in plugin_cls.preflight_readiness(exp_dir=exp_root, cfg=cfg, reads=(step.reads or {})):
            issues.append((label, step.id, issue))
    return issues


def _source_record_preflight_issues(
    *,
    items: list[Any],
    registry: Any,
    contracts: Any,
) -> tuple[int, list[str]]:
    declared = 0
    issues: list[str] = []
    for step in items:
        plugin_cls = registry.resolve(step.plugin)
        ports = plugin_cls.input_ports()
        for label, ref in (step.reads or {}).items():
            if not isinstance(ref, RecordCollectionRef):
                continue
            declared += len(ref.records)
            if not ref.records:
                issues.append(f"pipeline:{step.id} • {label} → no source records declared")
                continue
            expected_contract = ports[label].contract
            for source_ref in ref.records:
                source_label = f"{source_ref.experiment_id}:{source_ref.record_id}"
                try:
                    resolved = resolve_source_record(source_ref, contracts=contracts)
                    record = resolved.record
                    if not isinstance(record, DataFrameArtifactRecord):
                        raise ConfigError(f"source record is {record.kind!r}, not a dataframe artifact")
                    if expected_contract is not None and not contracts.satisfies(
                        actual=record.contract_id,
                        expected=expected_contract,
                    ):
                        raise ConfigError(f"contract {record.contract_id!r} does not satisfy {expected_contract!r}")
                    resolved.verify_artifact_integrity()
                except (OSError, ReaderError) as exc:
                    issues.append(f"pipeline:{step.id} • {label} → {source_label}: {exc}")
    return declared, issues


def _planned_output_record_ids(
    *,
    pipeline_items: list[Any],
    plot_items: list[Any],
    export_items: list[Any],
    registry: Any,
) -> set[str]:
    planned = {f"plot:{step.id}" for step in plot_items}
    planned.update(f"export:{step.id}" for step in export_items)
    for step in pipeline_items:
        plugin_cls = registry.resolve(step.plugin)
        planned.update(
            ref.record_id
            for ref in _resolve_output_labels(
                step_id=step.id,
                output_ports=plugin_cls.output_ports(),
                writes=(step.writes or {}),
            ).values()
        )
    return planned


def _assert_no_source_record_output_collisions(
    *,
    items: list[Any],
    planned_record_ids: set[str],
    experiment_root: Path,
) -> None:
    for step in items:
        for label, ref in (step.reads or {}).items():
            if not isinstance(ref, RecordCollectionRef):
                continue
            for source_ref in ref.records:
                if source_ref.experiment_root == experiment_root and source_ref.record_id in planned_record_ids:
                    source_label = f"{source_ref.experiment_id}:{source_ref.record_id}"
                    raise ConfigError(
                        f"pipeline {step.id}: reads '{label}' source record {source_label!r} targets the same "
                        f"experiment and collides with planned output {source_ref.record_id!r}. Choose a source "
                        "owned by another experiment or a distinct output record id."
                    )


def validation_summary(
    decl: WorkbenchDecl,
    *,
    check_files: bool = False,
    exp_root: Path | None = None,
    runtime: ReaderRuntime | None = None,
    plot_specs_override: list[Any] | None = None,
    export_specs_override: list[Any] | None = None,
) -> dict[str, Any]:
    runtime = runtime or builtin_runtime()
    workbench = resolve_workbench(decl)
    pipeline_steps = list(workbench.pipeline)
    plot_specs = list(workbench.plots) if plot_specs_override is None else list(plot_specs_override)
    export_specs = list(workbench.exports) if export_specs_override is None else list(export_specs_override)
    notebook_specs = list(workbench.notebooks)
    ensure_unique_workbench_ids(pipeline_steps, plot_specs, export_specs, notebook_specs)
    categories = collect_categories(list(workbench.plugin_steps()))
    if "plot" in categories:
        ensure_mpl_cache_dir()
    registry = runtime.plugins if categories else None
    bound_protocol = runtime.bind_protocol(decl.experiment_semantics.protocol)

    experiment_semantics = decl.experiment_semantics
    pipeline_labels = _validate_pipeline_steps(
        pipeline_steps,
        registry=registry,
        experiment=experiment_semantics,
        protocol=bound_protocol,
    )
    if registry is not None:
        planned_record_ids = _planned_output_record_ids(
            pipeline_items=pipeline_steps,
            plot_items=plot_specs,
            export_items=export_specs,
            registry=registry,
        )
        _assert_no_source_record_output_collisions(
            items=pipeline_steps,
            planned_record_ids=planned_record_ids,
            experiment_root=decl.experiment.root,
        )
    if plot_specs:
        if registry is None:
            raise ConfigError("plot validation requires plugin-backed workbench specs")
        _validate_specs(
            plot_specs,
            label="plot",
            registry=registry,
            available_labels=pipeline_labels,
            protocol=bound_protocol,
            experiment=experiment_semantics,
        )
    if export_specs:
        if registry is None:
            raise ConfigError("export validation requires plugin-backed workbench specs")
        _validate_specs(
            export_specs,
            label="export",
            registry=registry,
            available_labels=pipeline_labels,
            protocol=bound_protocol,
        )
    if notebook_specs:
        _validate_notebook_specs(notebook_specs, protocol=bound_protocol)

    declared_entries: list[tuple[str, str, str, Path]] = []
    declared_roots: list[tuple[str, str, Path]] = []
    if exp_root is not None:
        for label, items in (("pipeline", pipeline_steps), ("plot", plot_specs), ("export", export_specs)):
            for step in items:
                for key, target in (step.reads or {}).items():
                    if isinstance(target, FileRef | ResourceRef):
                        declared_entries.append(
                            (label, step.id, key, _resolve_exp_path(target.path, exp_root=exp_root))
                        )
                with_block = _effective_step_with(protocol=bound_protocol, step=step) if hasattr(step, "with_") else {}
                auto_roots = with_block.get("auto_roots")
                if isinstance(auto_roots, list):
                    for root in auto_roots:
                        declared_roots.append((label, step.id, _resolve_exp_path(Path(root), exp_root=exp_root)))

    files_checked = None
    file_issues: list[str] = []
    dependency_issues: list[str] = []
    source_record_issues: list[str] = []
    source_records_declared = 0
    readiness_errors: list[str] = []
    if check_files:
        files_checked = (len(declared_entries), len(declared_roots))
        for label, step_id, key, path in declared_entries:
            if not path.exists():
                file_issues.append(f"{label}:{step_id} • {key} → {_render_rel(path, exp_root=exp_root)}")
        for label, step_id, root in declared_roots:
            if not root.exists():
                file_issues.append(f"{label}:{step_id} • auto_roots → {_render_rel(root, exp_root=exp_root)}")
        if registry is not None:
            source_records_declared, source_record_issues = _source_record_preflight_issues(
                items=pipeline_steps,
                registry=registry,
                contracts=runtime.contracts,
            )
            for label, step_id, issue in (
                _plugin_preflight_issues(
                    items=pipeline_steps,
                    label="pipeline",
                    registry=registry,
                    protocol=bound_protocol,
                    exp_root=exp_root,
                )
                + _plugin_preflight_issues(
                    items=plot_specs,
                    label="plot",
                    registry=registry,
                    protocol=bound_protocol,
                    exp_root=exp_root,
                )
                + _plugin_preflight_issues(
                    items=export_specs,
                    label="export",
                    registry=registry,
                    protocol=bound_protocol,
                    exp_root=exp_root,
                )
            ):
                rendered = f"{label}:{step_id} • {issue.message}"
                if issue.kind == "dependency":
                    dependency_issues.append(rendered)
                else:
                    file_issues.append(rendered)
        readiness_errors = [*file_issues, *dependency_issues, *source_record_issues]

    file_total, root_total = files_checked or (len(declared_entries), len(declared_roots))
    if not check_files:
        files_payload = {
            "mode": "skipped",
            "checked": False,
            "declared": {"file_inputs": file_total, "auto_roots": root_total},
            "issues": 0,
            "summary": "skipped (--no-files)",
        }
    elif file_total == 0 and root_total == 0 and not file_issues:
        files_payload = {
            "mode": "none_declared",
            "checked": True,
            "declared": {"file_inputs": 0, "auto_roots": 0},
            "issues": 0,
            "summary": "none declared",
        }
    else:
        parts = []
        if file_total:
            parts.append(f"{file_total} file input(s)")
        if root_total:
            parts.append(f"{root_total} auto_root(s)")
        if file_issues:
            parts.append(f"{len(file_issues)} issue(s)")
        files_payload = {
            "mode": "error" if file_issues else "ok",
            "checked": True,
            "declared": {"file_inputs": file_total, "auto_roots": root_total},
            "issues": len(file_issues),
            "summary": (f"error ({', '.join(parts)})" if file_issues else f"ok ({', '.join(parts)})"),
        }

    if not check_files:
        dependencies_payload = {
            "checked": False,
            "issues": 0,
            "summary": "skipped (--no-files)",
        }
    else:
        dependencies_payload = {
            "checked": True,
            "issues": len(dependency_issues),
            "summary": (f"error ({len(dependency_issues)} issue(s))" if dependency_issues else "ok"),
        }

    source_records_payload = {
        "checked": check_files,
        "declared": source_records_declared,
        "issues": len(source_record_issues),
        "summary": (
            "skipped (--no-files)"
            if not check_files
            else (
                f"error ({len(source_record_issues)} issue(s))"
                if source_record_issues
                else f"ok ({source_records_declared} record(s))"
            )
        ),
    }

    checks = [
        "schema",
        "plugin availability",
        "reads",
        "output labels",
        "plugin config",
    ]
    if check_files:
        checks.extend(("runtime readiness", "source record revisions"))

    return {
        "status": "error" if readiness_errors else "ok",
        "protocol": decl.experiment_semantics.protocol.id,
        "counts": {
            "pipeline": len(pipeline_steps),
            "plots": len(plot_specs),
            "exports": len(export_specs),
            "notebooks": len(notebook_specs),
        },
        "checks": checks,
        "files": files_payload,
        "dependencies": dependencies_payload,
        "source_records": source_records_payload,
        "errors": readiness_errors,
        "tip": (
            f"fix readiness issues or use '{reader_command('validate', '--no-files')}' for config-only checks"
            if readiness_errors
            else "use 'reader explain' to see inputs/outputs"
        ),
    }


def validate(
    decl: WorkbenchDecl,
    *,
    console: Console,
    check_files: bool = False,
    exp_root: Path | None = None,
    runtime: ReaderRuntime | None = None,
) -> dict[str, Any]:
    summary = validation_summary(
        decl,
        check_files=check_files,
        exp_root=exp_root,
        runtime=runtime,
    )
    ok = summary["status"] == "ok"
    lines = [
        "[green]✓ Config validated[/green]" if ok else "[error]✗ Validation failed[/error]",
        f"[dim]protocol[/dim]: {summary['protocol']}",
        f"[dim]pipeline[/dim]: {summary['counts']['pipeline']}",
        f"[dim]plots[/dim]: {summary['counts']['plots']}",
        f"[dim]exports[/dim]: {summary['counts']['exports']}",
        f"[dim]notebooks[/dim]: {summary['counts']['notebooks']}",
        f"[dim]checks[/dim]: {', '.join(summary['checks'])}",
    ]
    lines.append(f"[dim]files[/dim]: {summary['files']['summary']}")
    lines.append(f"[dim]dependencies[/dim]: {summary['dependencies']['summary']}")
    lines.append(f"[dim]source records[/dim]: {summary['source_records']['summary']}")
    if summary["errors"]:
        lines.append("[dim]errors[/dim]:")
        lines.extend(f"- {item}" for item in summary["errors"])
    lines.append(f"[dim]tip[/dim]: {summary['tip']}")
    console.print(Panel.fit("\n".join(lines), border_style=("green" if ok else "error"), box=box.ROUNDED))
    return summary
