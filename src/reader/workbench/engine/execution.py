from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from reader.errors import ExecutionError, ReaderError
from reader.workbench.context import RunContext
from reader.workbench.graph import (
    FileRef,
    ProvenanceInput,
    RecordCollectionRef,
    RecordRef,
    ResourceRef,
    SourceRecordRef,
)
from reader.workbench.records import PathDescription, RecordInputEvidence, RecordStore

from ._shared import digest_cfg
from .contracts import (
    _assert_input_ports,
    _assert_output_ports,
    _resolve_output_labels,
    _resolve_runtime_output_ports,
    collect_file_output_descriptions,
    collect_file_output_paths,
)
from .file_outputs import FileOutputTransaction
from .inputs import _resolve_inputs, resolve_missing_file_inputs


def _runtime_provenance_inputs(*, step: Any, inputs: dict[str, Any]) -> list[ProvenanceInput]:
    provenance: list[ProvenanceInput] = []
    declared_reads = dict(step.reads or {})
    for label in sorted(inputs):
        ref = declared_reads.get(label)
        value = inputs[label]
        if isinstance(ref, RecordCollectionRef):
            provenance.extend(
                ProvenanceInput(
                    label=f"{label}[{item.resource_id}]",
                    ref=item,
                    discovery_policy="source_record",
                )
                for item in ref.records
            )
            continue
        if isinstance(value, tuple) and value and all(isinstance(item, Path) for item in value):
            for index, path in enumerate(value):
                item_ref = ref if ref is not None and len(value) == 1 else FileRef(path=path)
                if isinstance(item_ref, ResourceRef):
                    policy = "declared_resource"
                elif ref is not None:
                    policy = "declared_file"
                else:
                    policy = "plugin_discovery"
                provenance.append(
                    ProvenanceInput(
                        label=f"{label}[{index}]",
                        ref=item_ref,
                        discovery_policy=policy,
                    )
                )
            continue
        if ref is None:
            if not isinstance(value, Path):
                raise ExecutionError(
                    f"{step.id}: resolved input {label!r} has no declared reference and is not a file path"
                )
            ref = FileRef(path=value)
            discovery_policy = "plugin_discovery"
        elif isinstance(ref, RecordRef):
            discovery_policy = "record"
        elif isinstance(ref, SourceRecordRef):
            discovery_policy = "source_record"
        elif isinstance(ref, ResourceRef):
            discovery_policy = "declared_resource"
        else:
            discovery_policy = "declared_file"
        provenance.append(ProvenanceInput(label=label, ref=ref, discovery_policy=discovery_policy))
    return provenance


def _persist_dataframe_outputs(
    *,
    store: RecordStore,
    ctx: RunContext,
    step: Any,
    cfg: Any,
    input_evidence: tuple[RecordInputEvidence, ...],
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
                inputs=input_evidence,
                config_digest=ctx.config_digest,
                producer_config_digest=digest_cfg(cfg),
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
    input_evidence: tuple[RecordInputEvidence, ...],
    outputs: dict[str, Any],
    resolved_output_ports: dict[str, Any],
    phase: str,
    description: str,
    protocol_figure_description: str | None,
) -> None:
    explicit_files = collect_file_output_paths(
        output_ports=resolved_output_ports,
        outputs=outputs,
        where=f"{phase}:{step.id}",
    )
    if not explicit_files:
        raise ExecutionError(f"{phase} {step.id}: must emit at least one explicit file output")
    producer_kind = "plot" if phase == "plots" else "export"
    record_files = sorted(
        {_record_output_path(path, outputs_dir=ctx.outputs_dir) for path in explicit_files},
        key=str,
    )
    path_descriptions: tuple[PathDescription, ...] = ()
    if phase == "plots":
        explicit_descriptions = collect_file_output_descriptions(
            output_ports=resolved_output_ports,
            outputs=outputs,
        )
        if protocol_figure_description is not None:
            path_descriptions = tuple(
                PathDescription(path=path, description=protocol_figure_description) for path in record_files
            )
        else:
            explicit_by_path: dict[Path, str] = {}
            for item in explicit_descriptions:
                record_path = _record_output_path(item.path, outputs_dir=ctx.outputs_dir)
                if record_path in explicit_by_path:
                    raise ExecutionError(f"plots {step.id}: duplicate descriptions for {record_path}")
                explicit_by_path[record_path] = item.description
            path_descriptions = tuple(
                PathDescription(path=path, description=explicit_by_path.get(path, description)) for path in record_files
            )
    store.append_file_bundle(
        producer_kind=producer_kind,
        producer_id=step.id,
        producer_plugin=step.plugin,
        record_id=f"{producer_kind}:{step.id}",
        inputs=input_evidence,
        config_digest=ctx.config_digest,
        producer_config_digest=digest_cfg(cfg),
        files=record_files,
        description=description,
        path_descriptions=path_descriptions,
        source_recipe=step.source_recipe,
    )


def _record_output_path(path: Path, *, outputs_dir: Path) -> Path:
    try:
        return path.relative_to(outputs_dir)
    except ValueError:
        return path


def _protocol_figure_description(*, ctx: RunContext, step: Any) -> str | None:
    for figure in ctx.protocol.descriptor.figures:
        if figure.id == step.id:
            return figure.summary
    return None


def execute_step(*, step: Any, phase: str, store: RecordStore, ctx: RunContext, registry: Any) -> None:
    descriptor = registry.resolve_descriptor(step.plugin)
    plugin_cls = descriptor.cls
    if ctx.protocol is None:
        raise ExecutionError(f"{phase} {step.id}: run context is missing a bound protocol")
    effective_with = ctx.protocol.effective_plugin_config(plugin_id=step.plugin, step_with=(step.with_ or {}))
    cfg = plugin_cls.ConfigModel.model_validate(effective_with)
    plugin = plugin_cls()
    plugin.bind_runtime(descriptor=descriptor, contracts=registry.contracts)
    input_ports = plugin.input_ports()
    output_ports = plugin.output_ports()
    output_labels = _resolve_output_labels(
        step_id=step.id,
        output_ports=output_ports,
        writes=(step.writes or {}),
    )
    inputs = _resolve_inputs(store, step.reads or {}, input_ports=input_ports, exp_dir=ctx.exp_dir)
    inputs = resolve_missing_file_inputs(
        plugin=plugin,
        exp_dir=ctx.exp_dir,
        cfg=cfg,
        inputs=inputs,
        input_ports=input_ports,
    )
    _assert_input_ports(
        plugin,
        inputs,
        contracts=registry.contracts,
        where=step.id,
    )
    provenance_inputs = _runtime_provenance_inputs(step=step, inputs=inputs)
    input_evidence = store.capture_inputs(provenance_inputs, resolved_inputs=inputs)

    plug_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "load_dataframe"):
            plug_inputs[key] = value.load_dataframe()
        else:
            plug_inputs[key] = value

    transaction_context = (
        FileOutputTransaction(context=ctx, step_id=step.id, phase=phase)
        if phase in {"plots", "exports"}
        else nullcontext()
    )
    with transaction_context as file_transaction:
        plugin_context = file_transaction.context if file_transaction is not None else ctx
        try:
            outputs = plugin.run(plugin_context, plug_inputs, cfg)
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
        )
        _persist_dataframe_outputs(
            store=store,
            ctx=ctx,
            step=step,
            cfg=cfg,
            input_evidence=input_evidence,
            outputs=outputs,
            resolved_output_ports=resolved_output_ports,
            output_labels=output_labels,
            phase=phase,
        )
        if phase in {"plots", "exports"}:
            outputs = file_transaction.promote(
                outputs=outputs,
                output_ports=resolved_output_ports,
                where=f"{phase}:{step.id}",
            )
            protocol_figure_description = _protocol_figure_description(ctx=ctx, step=step) if phase == "plots" else None
            description = protocol_figure_description or descriptor.summary
            _persist_file_bundle_record(
                store=store,
                ctx=ctx,
                step=step,
                cfg=cfg,
                input_evidence=input_evidence,
                outputs=outputs,
                resolved_output_ports=resolved_output_ports,
                phase=phase,
                description=description,
                protocol_figure_description=protocol_figure_description,
            )
            file_transaction.commit()


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
