from __future__ import annotations

from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel

from reader.errors import ConfigError, ExecutionError
from reader.plotting.mpl import ensure_mpl_cache_dir
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.experiment import ExperimentSemantics
from reader.workbench.graph import (
    FileRef,
    RecordRef,
    ResourceRef,
    ensure_unique_workbench_ids,
    input_ref_display,
    resolve_workbench,
)
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
        if port.kind == "dataframe" and isinstance(target, FileRef | ResourceRef):
            raise ConfigError(
                f"{label} {step.id}: reads '{key}' targets '{input_ref_display(target)}', "
                "but the plugin expects a dataframe record."
            )
        if port.kind == "file_path" and isinstance(target, RecordRef):
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
        if isinstance(with_block.get("logic_map_ref"), str):
            experiment.annotations.resolve_logic_map(ref=with_block["logic_map_ref"])
    except Exception as err:
        raise ConfigError(f"plot {step.id}: invalid ordering semantic reference: {err}") from err


def _validate_pipeline_semantic_refs(*, step: Any, experiment: ExperimentSemantics, with_block: dict[str, Any]) -> None:
    try:
        if isinstance(with_block.get("logic_map_ref"), str):
            experiment.annotations.resolve_logic_map(ref=with_block["logic_map_ref"])
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


def validation_summary(
    decl: WorkbenchDecl,
    *,
    check_files: bool = False,
    exp_root: Path | None = None,
    runtime: ReaderRuntime | None = None,
) -> dict[str, Any]:
    runtime = runtime or builtin_runtime()
    workbench = resolve_workbench(decl)
    pipeline_steps = list(workbench.pipeline)
    plot_specs = list(workbench.plots)
    export_specs = list(workbench.exports)
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

    files_checked = None
    missing_files: list[tuple[str, str, str, Path]] = []
    missing_roots: list[tuple[str, str, Path]] = []
    if check_files:
        entries: list[tuple[str, str, str, Path]] = []
        roots: list[tuple[str, str, Path]] = []
        for label, items in (("pipeline", pipeline_steps), ("plot", plot_specs), ("export", export_specs)):
            for step in items:
                for key, target in (step.reads or {}).items():
                    if isinstance(target, FileRef | ResourceRef):
                        entries.append((label, step.id, key, target.path))
                with_block = _effective_step_with(protocol=bound_protocol, step=step) if hasattr(step, "with_") else {}
                auto_roots = with_block.get("auto_roots")
                if isinstance(auto_roots, list):
                    for root in auto_roots:
                        roots.append((label, step.id, Path(root)))
        files_checked = (len(entries), len(roots))
        for label, step_id, key, path in entries:
            if not path.exists():
                missing_files.append((label, step_id, key, path))
        for label, step_id, root in roots:
            if not root.exists():
                missing_roots.append((label, step_id, root))
        if missing_files or missing_roots:
            lines = ["Missing input files:"]
            for label, step_id, key, path in missing_files:
                rel = path
                if exp_root is not None:
                    try:
                        rel = path.relative_to(exp_root)
                    except Exception:
                        rel = path
                lines.append(f"- {label}:{step_id} • {key} → {rel}")
            for label, step_id, root in missing_roots:
                rel = root
                if exp_root is not None:
                    try:
                        rel = root.relative_to(exp_root)
                    except Exception:
                        rel = root
                lines.append(f"- {label}:{step_id} • auto_roots → {rel}")
            raise ConfigError("\n".join(lines))

    file_total, root_total = files_checked or (0, 0)
    if not check_files:
        files_payload = {
            "mode": "skipped",
            "checked": False,
            "declared": {"file_inputs": 0, "auto_roots": 0},
            "summary": "skipped (--no-files)",
        }
    elif file_total == 0 and root_total == 0:
        files_payload = {
            "mode": "none_declared",
            "checked": True,
            "declared": {"file_inputs": 0, "auto_roots": 0},
            "summary": "none declared",
        }
    else:
        parts = []
        if file_total:
            parts.append(f"{file_total} file input(s)")
        if root_total:
            parts.append(f"{root_total} auto_root(s)")
        files_payload = {
            "mode": "ok",
            "checked": True,
            "declared": {"file_inputs": file_total, "auto_roots": root_total},
            "summary": f"ok ({', '.join(parts)})",
        }

    return {
        "status": "ok",
        "protocol": decl.experiment_semantics.protocol.id,
        "counts": {
            "pipeline": len(pipeline_steps),
            "plots": len(plot_specs),
            "exports": len(export_specs),
            "notebooks": len(notebook_specs),
        },
        "checks": [
            "schema",
            "plugin availability",
            "reads",
            "output labels",
            "plugin config",
        ],
        "files": files_payload,
        "tip": "use 'reader explain' to see inputs/outputs",
    }


def validate(
    decl: WorkbenchDecl,
    *,
    console: Console,
    check_files: bool = False,
    exp_root: Path | None = None,
    runtime: ReaderRuntime | None = None,
) -> None:
    summary = validation_summary(
        decl,
        check_files=check_files,
        exp_root=exp_root,
        runtime=runtime,
    )
    lines = [
        "[green]✓ Config validated[/green]",
        f"[dim]protocol[/dim]: {summary['protocol']}",
        f"[dim]pipeline[/dim]: {summary['counts']['pipeline']}",
        f"[dim]plots[/dim]: {summary['counts']['plots']}",
        f"[dim]exports[/dim]: {summary['counts']['exports']}",
        f"[dim]notebooks[/dim]: {summary['counts']['notebooks']}",
        f"[dim]checks[/dim]: {', '.join(summary['checks'])}",
    ]
    lines.append(f"[dim]files[/dim]: {summary['files']['summary']}")
    lines.append(f"[dim]tip[/dim]: {summary['tip']}")
    console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))
