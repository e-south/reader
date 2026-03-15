from __future__ import annotations

from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel

from reader.core.config import ReaderSpec
from reader.core.errors import ConfigError, ExecutionError
from reader.core.mpl import ensure_mpl_cache_dir
from reader.core.notebooks import resolve_notebook_template_descriptor
from reader.core.registry import load_entry_points
from reader.core.semantics import resolve_assay_order_arg, resolve_logic_map_ref, resolve_plot_partition
from reader.core.workbench import ensure_unique_workbench_ids, resolve_workbench

from ._shared import collect_categories
from .contracts import _resolve_output_labels


def _validate_reads_shape(*, label: str, step: Any, plugin_cls: Any) -> None:
    try:
        required = plugin_cls.input_contracts()
        expected = set()
        mandatory = set()
        for raw_name in required:
            optional = raw_name.endswith("?")
            name = raw_name[:-1] if optional else raw_name
            expected.add(name)
            if not optional:
                mandatory.add(name)
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


def _assert_known_read_targets(*, label: str, step: Any, available_labels: set[str]) -> None:
    for key, target in (step.reads or {}).items():
        if isinstance(target, str) and target.startswith("file:"):
            continue
        if target in available_labels:
            continue
        preview = sorted(available_labels)
        shown = ", ".join(preview[:12]) if preview else "—"
        tail = " …" if len(preview) > 12 else ""
        source = "pipeline" if label in {"plot", "export"} else "any prior step"
        raise ConfigError(
            f"{label} {step.id}: reads '{key}' → '{target}', which is not produced by {source}. "
            f"Known labels{' so far' if label == 'pipeline' else ''}: {shown}{tail}. "
            "Check writes/reads aliases or use file: paths."
        )


def _validate_step_config(*, label: str, step: Any, plugin_cls: Any) -> None:
    try:
        plugin_cls.ConfigModel.model_validate(step.with_ or {})
    except Exception as err:
        raise ConfigError(f"{label} {step.id}: invalid config for {step.uses}: {err}") from err


def _validate_plot_semantic_refs(*, step: Any, assay: dict[str, Any]) -> None:
    with_block = step.with_ or {}
    partition = with_block.get("partition")
    if partition is not None:
        try:
            resolve_plot_partition(partition=partition, assay=assay)
        except Exception as err:
            raise ConfigError(f"plot {step.id}: invalid plot partition: {err}") from err

    x_column = with_block.get("x")
    y_column = with_block.get("y")
    try:
        resolve_assay_order_arg(
            order=with_block.get("order_x"),
            order_ref=with_block.get("order_x_ref"),
            column=(x_column if isinstance(x_column, str) else None),
            assay=assay,
            arg_name="order_x",
        )
        resolve_assay_order_arg(
            order=with_block.get("order_y"),
            order_ref=with_block.get("order_y_ref"),
            column=(y_column if isinstance(y_column, str) else None),
            assay=assay,
            arg_name="order_y",
        )
        if isinstance(with_block.get("logic_map_ref"), str):
            resolve_logic_map_ref(ref=with_block["logic_map_ref"], assay=assay)
    except Exception as err:
        raise ConfigError(f"plot {step.id}: invalid ordering semantic reference: {err}") from err


def _validate_pipeline_semantic_refs(*, step: Any, assay: dict[str, Any]) -> None:
    with_block = step.with_ or {}
    try:
        if isinstance(with_block.get("logic_map_ref"), str):
            resolve_logic_map_ref(ref=with_block["logic_map_ref"], assay=assay)
    except Exception as err:
        raise ConfigError(f"pipeline {step.id}: invalid assay semantic reference: {err}") from err


def _validate_pipeline_steps(items: list[Any], *, registry: Any, assay: dict[str, Any]) -> set[str]:
    available: set[str] = set()
    produced: set[str] = set()
    for step in items:
        try:
            plugin_cls = registry.resolve(step.uses)
        except Exception as err:
            raise ConfigError(f"pipeline {step.id}: {err}") from err
        _validate_reads_shape(label="pipeline", step=step, plugin_cls=plugin_cls)
        _assert_known_read_targets(label="pipeline", step=step, available_labels=available | produced)
        _validate_step_config(label="pipeline", step=step, plugin_cls=plugin_cls)
        _validate_pipeline_semantic_refs(step=step, assay=assay)
        try:
            output_labels = _resolve_output_labels(
                step_id=step.id,
                output_contracts=plugin_cls.output_contracts(),
                writes=(step.writes or {}),
            )
        except ExecutionError as err:
            raise ConfigError(f"pipeline {step.id}: {err}") from err
        for out_name, out_label in output_labels.items():
            if plugin_cls.output_contracts()[out_name] == "none":
                continue
            if out_label in available or out_label in produced:
                raise ConfigError(
                    f"pipeline {step.id}: output label '{out_label}' is already produced by another step. "
                    "Use a unique writes mapping to avoid clobbering dataframe records."
                )
            produced.add(out_label)
    return available | produced


def _validate_specs(
    items: list[Any],
    *,
    label: str,
    registry: Any,
    available_labels: set[str],
    assay: dict[str, Any] | None = None,
) -> None:
    for spec_item in items:
        try:
            plugin_cls = registry.resolve(spec_item.uses)
        except Exception as err:
            raise ConfigError(f"{label} {spec_item.id}: {err}") from err
        _validate_reads_shape(label=label, step=spec_item, plugin_cls=plugin_cls)
        _assert_known_read_targets(label=label, step=spec_item, available_labels=available_labels)
        _validate_step_config(label=label, step=spec_item, plugin_cls=plugin_cls)
        if label == "plot":
            _validate_plot_semantic_refs(
                step=spec_item,
                assay=dict(assay or {}),
            )
        data_outputs = {key: value for key, value in plugin_cls.output_contracts().items() if value != "none"}
        if data_outputs:
            raise ConfigError(
                f"{label} {spec_item.id}: plugins must not declare data outputs (got {data_outputs}). "
                "Move data outputs into the pipeline."
            )


def _validate_notebook_specs(items: list[Any]) -> None:
    for spec_item in items:
        if spec_item.reads:
            raise ConfigError(f"notebook {spec_item.id}: notebook specs must not declare reads.")
        if spec_item.writes:
            raise ConfigError(f"notebook {spec_item.id}: notebook specs must not declare writes.")
        resolve_notebook_template_descriptor(spec_item.uses)


def validate(spec: ReaderSpec, *, console: Console, check_files: bool = False, exp_root: Path | None = None) -> None:
    workbench = resolve_workbench(spec)
    pipeline_steps = list(workbench.pipeline)
    plot_specs = list(workbench.plots)
    export_specs = list(workbench.exports)
    notebook_specs = list(workbench.notebooks)
    ensure_unique_workbench_ids(pipeline_steps, plot_specs, export_specs, notebook_specs)
    categories = collect_categories(list(workbench.plugin_specs()))
    if "plot" in categories:
        ensure_mpl_cache_dir()
    registry = load_entry_points(categories=categories) if categories else None

    assay = {
        "labels": dict(spec.assay.labels or {}),
        "orders": dict(spec.assay.orders or {}),
        "collections": dict(spec.assay.collections or {}),
        "logic_maps": dict(spec.assay.logic_maps or {}),
    }
    pipeline_labels = _validate_pipeline_steps(pipeline_steps, registry=registry, assay=assay)
    if plot_specs:
        if registry is None:
            raise ConfigError("plot validation requires plugin-backed workbench specs")
        _validate_specs(
            plot_specs,
            label="plot",
            registry=registry,
            available_labels=pipeline_labels,
            assay=assay,
        )
    if export_specs:
        if registry is None:
            raise ConfigError("export validation requires plugin-backed workbench specs")
        _validate_specs(export_specs, label="export", registry=registry, available_labels=pipeline_labels)
    if notebook_specs:
        _validate_notebook_specs(notebook_specs)

    files_checked = None
    missing_files: list[tuple[str, str, str, Path]] = []
    missing_roots: list[tuple[str, str, Path]] = []
    if check_files:
        entries: list[tuple[str, str, str, Path]] = []
        roots: list[tuple[str, str, Path]] = []
        for label, items in (("pipeline", pipeline_steps), ("plot", plot_specs), ("export", export_specs)):
            for step in items:
                for key, target in (step.reads or {}).items():
                    if isinstance(target, str) and target.startswith("file:"):
                        entries.append((label, step.id, key, Path(target.split("file:", 1)[1])))
                auto_roots = (step.with_ or {}).get("auto_roots") if hasattr(step, "with_") else None
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

    lines = [
        "[green]✓ Config validated[/green]",
        f"[dim]pipeline[/dim]: {len(pipeline_steps)}",
        f"[dim]plots[/dim]: {len(plot_specs)}",
        f"[dim]exports[/dim]: {len(export_specs)}",
        f"[dim]notebooks[/dim]: {len(notebook_specs)}",
        "[dim]checks[/dim]: schema, plugin availability, reads, output labels, plugin config",
    ]
    if check_files:
        file_total, root_total = files_checked or (0, 0)
        if file_total == 0 and root_total == 0:
            lines.append("[dim]files[/dim]: none declared")
        else:
            parts = []
            if file_total:
                parts.append(f"{file_total} file input(s)")
            if root_total:
                parts.append(f"{root_total} auto_root(s)")
            lines.append(f"[dim]files[/dim]: ok ({', '.join(parts)})")
    else:
        lines.append("[dim]files[/dim]: skipped (--no-files)")
    lines.append("[dim]tip[/dim]: use 'reader explain' to see inputs/outputs")
    console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))
