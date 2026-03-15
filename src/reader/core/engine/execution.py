from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from reader.core.context import RunContext
from reader.core.errors import ExecutionError, ReaderError
from reader.core.records import RecordStore

from ._shared import diff_files, digest_cfg, snapshot_dir
from .contracts import (
    _assert_input_contracts,
    _assert_output_contracts,
    _resolve_output_labels,
    _resolve_runtime_output_contracts,
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
    resolved_output_contracts: dict[str, str],
    output_labels: dict[str, str],
    phase: str,
) -> None:
    for out_name, obj in outputs.items():
        contract_id = resolved_output_contracts[out_name]
        if isinstance(obj, pd.DataFrame) and contract_id != "none":
            store.persist_dataframe(
                producer_id=step.id,
                producer_uses=step.uses,
                out_name=out_name,
                record_id=output_labels[out_name],
                df=obj,
                contract_id=contract_id,
                inputs=[
                    inputs[name].record_id if hasattr(inputs[name], "record_id") else str(inputs[name])
                    for name in (step.reads or {})
                ],
                config_digest=digest_cfg(cfg),
                validate_contract=strict,
            )
            continue
        if contract_id == "none":
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
    phase: str,
    pre_state: dict[Path, float],
) -> None:
    base_dir = ctx.plots_dir if phase == "plots" else ctx.exports_dir
    post_state = snapshot_dir(base_dir)
    changed = diff_files(pre_state, post_state)
    file_outputs = outputs.get("files")
    extra_files: list[Path] = []
    if file_outputs:
        if isinstance(file_outputs, str | Path):
            extra_files = [Path(file_outputs)]
        elif isinstance(file_outputs, list):
            extra_files = [Path(path) for path in file_outputs if path]
    combined = list({*changed, *extra_files})
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
        producer_uses=step.uses,
        record_id=f"{producer_kind}:{step.id}",
        inputs=[
            inputs[name].record_id if hasattr(inputs[name], "record_id") else str(inputs[name])
            for name in (step.reads or {})
        ],
        config_digest=digest_cfg(cfg),
        files=[Path(path) for path in sorted(set(rel_files))],
    )


def execute_step(*, step: Any, phase: str, store: RecordStore, ctx: RunContext, registry: Any) -> None:
    plugin_cls = registry.resolve(step.uses)
    cfg = plugin_cls.ConfigModel.model_validate(step.with_ or {})
    plugin = plugin_cls()
    output_contracts = plugin.output_contracts()
    output_labels = _resolve_output_labels(
        step_id=step.id,
        output_contracts=output_contracts,
        writes=(step.writes or {}),
    )
    pre_state: dict[Path, float] = {}
    if phase == "plots":
        pre_state = snapshot_dir(ctx.plots_dir)
    elif phase == "exports":
        pre_state = snapshot_dir(ctx.exports_dir)

    inputs = _resolve_inputs(store, step.reads or {}, exp_dir=ctx.exp_dir)
    _assert_input_contracts(plugin, inputs, where=step.id, strict=ctx.strict, logger=ctx.logger)

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

    resolved_output_contracts = _resolve_runtime_output_contracts(
        plugin,
        inputs=inputs,
        outputs=outputs,
        cfg=cfg,
        where=step.id,
    )
    _assert_output_contracts(
        resolved_output_contracts,
        outputs,
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
        resolved_output_contracts=resolved_output_contracts,
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
            ctx.logger.info("→ %s %s [%d/%d] uses=%s", phase, step.id, ordinal, len(items), step.uses)
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
