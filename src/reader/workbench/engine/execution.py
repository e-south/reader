from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from reader.core.errors import ExecutionError, ReaderError
from reader.workbench.context import RunContext
from reader.workbench.graph import ProvenanceInput
from reader.workbench.records import RecordStore

from ._shared import diff_files, digest_cfg, snapshot_dir
from .contracts import (
    _assert_input_ports,
    _assert_output_ports,
    _resolve_output_labels,
    _resolve_runtime_output_ports,
    collect_file_output_paths,
)
from .inputs import _resolve_inputs


def _persist_dataframe_outputs(
    *,
    store: RecordStore,
    step: Any,
    cfg: Any,
    strict: bool,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    resolved_output_ports: dict[str, Any],
    output_labels: dict[str, Any],
    phase: str,
) -> None:
    for out_name, obj in outputs.items():
        port = resolved_output_ports[out_name]
        if port.kind != "dataframe":
            continue
        if isinstance(obj, pd.DataFrame):
            store.persist_dataframe(
                producer_id=step.id,
                producer_kind="pipeline" if phase == "pipeline" else ("plot" if phase == "plots" else "export"),
                producer_plugin=step.plugin,
                out_name=out_name,
                record_id=output_labels[out_name].record_id,
                df=obj,
                contract_id=port.contract or "",
                inputs=[ProvenanceInput(label=name, ref=step.reads[name]) for name in (step.reads or {})],
                config_digest=digest_cfg(cfg),
                validate_contract=strict,
                source_recipe=step.source_recipe,
            )
            continue
        raise ExecutionError(f"{phase} {step.id}: unsupported output type for {out_name}")


def _persist_file_bundle_record(
    *,
    store: RecordStore,
    ctx: RunContext,
    step: Any,
    cfg: Any,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    resolved_output_ports: dict[str, Any],
    phase: str,
    pre_state: dict[Path, float],
) -> None:
    base_dir = ctx.plots_dir if phase == "plots" else ctx.exports_dir
    post_state = snapshot_dir(base_dir)
    changed = diff_files(pre_state, post_state)
    explicit_files = collect_file_output_paths(
        output_ports=resolved_output_ports,
        outputs=outputs,
        where=f"{phase}:{step.id}",
    )
    combined = list({*changed, *explicit_files})
    if not combined:
        return
    rel_files: list[str] = []
    for path in combined:
        try:
            rel_files.append(str(path.relative_to(ctx.outputs_dir)))
        except Exception:
            rel_files.append(str(path))
    producer_kind = "plot" if phase == "plots" else "export"
    store.append_file_bundle(
        producer_kind=producer_kind,
        producer_id=step.id,
        producer_plugin=step.plugin,
        record_id=f"{producer_kind}:{step.id}",
        inputs=[ProvenanceInput(label=name, ref=step.reads[name]) for name in (step.reads or {})],
        config_digest=digest_cfg(cfg),
        files=[Path(path) for path in sorted(set(rel_files))],
        source_recipe=step.source_recipe,
    )


def execute_step(*, step: Any, phase: str, store: RecordStore, ctx: RunContext, registry: Any) -> None:
    descriptor = registry.resolve_descriptor(step.plugin)
    plugin_cls = descriptor.cls
    cfg = plugin_cls.ConfigModel.model_validate(step.with_ or {})
    plugin = plugin_cls()
    plugin.bind_runtime(descriptor=descriptor, contracts=registry.contracts)
    input_ports = plugin.input_ports()
    output_ports = plugin.output_ports()
    output_labels = _resolve_output_labels(
        step_id=step.id,
        output_ports=output_ports,
        writes=(step.writes or {}),
    )
    pre_state: dict[Path, float] = {}
    if phase == "plots":
        pre_state = snapshot_dir(ctx.plots_dir)
    elif phase == "exports":
        pre_state = snapshot_dir(ctx.exports_dir)

    inputs = _resolve_inputs(store, step.reads or {}, input_ports=input_ports, exp_dir=ctx.exp_dir)
    _assert_input_ports(
        plugin,
        inputs,
        contracts=registry.contracts,
        where=step.id,
        strict=ctx.strict,
        logger=ctx.logger,
    )

    plug_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "load_dataframe"):
            plug_inputs[key] = value.load_dataframe()
        else:
            plug_inputs[key] = value

    try:
        outputs = plugin.run(ctx, plug_inputs, cfg)
    except ReaderError:
        raise
    except Exception as err:
        raise ExecutionError(f"{phase} {step.id} crashed: {err}") from err

    resolved_output_ports = _resolve_runtime_output_ports(
        plugin,
        inputs=inputs,
        outputs=outputs,
        cfg=cfg,
        contracts=registry.contracts,
        where=step.id,
    )
    _assert_output_ports(
        resolved_output_ports,
        outputs,
        contracts=registry.contracts,
        where=step.id,
        strict=ctx.strict,
        logger=ctx.logger,
    )
    _persist_dataframe_outputs(
        store=store,
        step=step,
        cfg=cfg,
        strict=ctx.strict,
        inputs=inputs,
        outputs=outputs,
        resolved_output_ports=resolved_output_ports,
        output_labels=output_labels,
        phase=phase,
    )
    if phase in {"plots", "exports"}:
        _persist_file_bundle_record(
            store=store,
            ctx=ctx,
            step=step,
            cfg=cfg,
            inputs=inputs,
            outputs=outputs,
            resolved_output_ports=resolved_output_ports,
            phase=phase,
            pre_state=pre_state,
        )


def run_steps(
    *,
    items: list[Any],
    phase: str,
    verbose: bool,
    console: Console,
    ctx: RunContext,
    store: RecordStore,
    registry: Any,
) -> None:
    if not items:
        return
    if verbose:
        for ordinal, step in enumerate(items, 1):
            ctx.logger.info("→ %s %s [%d/%d] plugin=%s", phase, step.id, ordinal, len(items), step.plugin)
            execute_step(step=step, phase=phase, store=store, ctx=ctx, registry=registry)
        return

    with Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"{phase.title()} ({len(items)} steps)", total=len(items))
        for step in items:
            progress.update(task, description=f"{phase}: {step.id}")
            execute_step(step=step, phase=phase, store=store, ctx=ctx, registry=registry)
            progress.advance(task)
