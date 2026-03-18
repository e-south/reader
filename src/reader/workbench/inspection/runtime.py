from __future__ import annotations

from copy import deepcopy

from reader.runtime import ReaderRuntime
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.graph import OutputRef, input_ref_display, output_ref_to_dict, resolve_workbench


def plot_output_summaries(bound_protocol) -> dict[str, str]:
    return {item.id: item.summary for item in bound_protocol.descriptor.figures}


def export_output_summaries(bound_protocol) -> dict[str, str]:
    return {item.id: item.summary for item in bound_protocol.descriptor.artifacts}


def selected_plan_payload(*, spec: ReaderSpec, decl: WorkbenchDecl, runtime: ReaderRuntime) -> dict[str, object]:
    bound_protocol = runtime.bind_protocol(decl.experiment_semantics.protocol)
    workbench = resolve_workbench(decl)
    plot_ids = [spec_decl.id for spec_decl in workbench.plots]
    export_ids = [spec_decl.id for spec_decl in workbench.exports]
    notebook_templates = [notebook.template for notebook in workbench.notebooks]
    pipeline_ids = [step.id for step in workbench.pipeline]
    return {
        "plot_profile": spec.protocol.outputs.plots.profile or bound_protocol.default_plot_profile or "—",
        "notebook_template": spec.protocol.outputs.notebook.template or bound_protocol.default_notebook_template or "—",
        "pipeline": {"count": len(pipeline_ids), "ids": pipeline_ids},
        "plots": {"count": len(plot_ids), "ids": plot_ids},
        "exports": {"count": len(export_ids), "ids": export_ids},
        "notebooks": {"count": len(notebook_templates), "templates": notebook_templates},
    }


def selected_plan_summary(selected: dict[str, object] | None) -> str:
    if not selected:
        return "—"
    pipeline = dict(selected["pipeline"])
    plots = dict(selected["plots"])
    exports = dict(selected["exports"])
    notebooks = dict(selected["notebooks"])
    profile = str(selected.get("plot_profile") or "—")
    return f"{profile} • {pipeline['count']} st • {plots['count']} pl • {exports['count']} ex • {notebooks['count']} nb"


def generated_summary(generated: dict[str, int]) -> str:
    return (
        f"{generated['records']} rec • "
        f"{generated['plots']} pl • "
        f"{generated['exports']} ex • "
        f"{generated['notebooks']} nb"
    )


def implementation_plan_payload(
    *,
    bound_protocol,
    decl: WorkbenchDecl,
    pipeline_steps,
    plot_steps,
    export_steps,
    notebook_steps,
) -> dict[str, object]:
    return {
        "protocol": bound_protocol.id,
        "input_sections": sorted(bound_protocol.inputs),
        "analysis_knobs": sorted(bound_protocol.analysis),
        "resources": sorted(decl.experiment_semantics.resources.by_id.keys()),
        "pipeline_flow": [step.id for step in pipeline_steps],
        "plots": [step.id for step in plot_steps],
        "exports": [step.id for step in export_steps],
        "notebooks": [step.template for step in notebook_steps],
    }


def compiled_workbench_payload(
    *,
    bound_protocol,
    pipeline_steps,
    plot_steps,
    export_steps,
    notebook_steps,
    runtime: ReaderRuntime,
    record_producers,
) -> dict[str, object]:
    plot_summaries = plot_output_summaries(bound_protocol)
    export_summaries = export_output_summaries(bound_protocol)
    return {
        "pipeline": [
            pipeline_step_payload(step, runtime=runtime, record_producers=record_producers) for step in pipeline_steps
        ],
        "plots": [
            spec_step_payload(
                step,
                summary=plot_summaries.get(step.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            for step in plot_steps
        ],
        "exports": [
            spec_step_payload(
                step,
                summary=export_summaries.get(step.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            for step in export_steps
        ],
        "notebooks": [{"id": step.id, "template": step.template} for step in notebook_steps],
    }


def record_producer_map(steps, *, runtime: ReaderRuntime) -> dict[str, dict[str, object]]:
    producers: dict[str, dict[str, object]] = {}
    for step in steps:
        plugin_cls = runtime.plugins.resolve_descriptor(step.plugin).cls
        for output_name, port in plugin_cls.output_ports().items():
            if port.kind != "dataframe":
                continue
            record_ref = (step.writes or {}).get(output_name, OutputRef(record_id=f"{step.id}/{output_name}"))
            producers[record_ref.record_id] = {
                "producer": {"id": step.id, "plugin": step.plugin, "stage": step.plugin.split("/", 1)[0]},
                "output": output_name,
                "contract": port.contract,
                "surface": _output_surface_payload(port),
            }
    return producers


def pipeline_step_payload(step, *, runtime: ReaderRuntime, record_producers=None) -> dict[str, object]:
    plugin_cls = runtime.plugins.resolve_descriptor(step.plugin).cls
    return {
        "stage": step.plugin.split("/", 1)[0],
        "id": step.id,
        "plugin": step.plugin,
        "semantics": plugin_semantics_payload(step.plugin, runtime=runtime),
        "reads": serialize_reads(
            step.reads, declared_ports=plugin_cls.input_ports(), record_producers=record_producers
        ),
        "writes": pipeline_writes_payload(step, runtime=runtime),
    }


def spec_step_payload(step, *, summary: str, runtime: ReaderRuntime, record_producers=None) -> dict[str, object]:
    plugin_cls = runtime.plugins.resolve_descriptor(step.plugin).cls
    return {
        "id": step.id,
        "plugin": step.plugin,
        "summary": summary,
        "semantics": plugin_semantics_payload(step.plugin, runtime=runtime),
        "reads": serialize_reads(
            step.reads, declared_ports=plugin_cls.input_ports(), record_producers=record_producers
        ),
    }


def render_read_binding(item: dict[str, object]) -> str:
    rendered = f"{item['label']} <- {item['display']}"
    source = item.get("source")
    if not isinstance(source, dict):
        return rendered
    surface = source.get("surface")
    if not isinstance(surface, dict):
        return rendered
    producer = source.get("producer")
    producer_id = producer.get("id") if isinstance(producer, dict) else None
    if not isinstance(producer_id, str) or not producer_id:
        return rendered
    mode = surface.get("runtime_mode")
    promoted = [str(value) for value in (surface.get("promoted") or []) if str(value).strip()]
    contract = item.get("contract")
    if mode == "promoted" and promoted:
        return f"{rendered} (via {producer_id}; may promote to {', '.join(promoted)})"
    if mode == "passthrough" and isinstance(contract, str) and contract in promoted:
        return f"{rendered} (via {producer_id}; preserves {contract})"
    return rendered


def plugin_semantics_payload(plugin: str, *, runtime: ReaderRuntime) -> dict[str, object]:
    descriptor = runtime.plugins.resolve_descriptor(plugin)
    return {
        "category": descriptor.category,
        "domain": descriptor.domain,
        "family": descriptor.family,
        "summary": descriptor.summary,
    }


def serialize_reads(reads, *, declared_ports=None, record_producers=None) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for label, ref in (reads or {}).items():
        item: dict[str, object] = {"label": label, "display": _binding_display(ref)}
        record_id = getattr(ref, "record_id", None)
        resource_id = getattr(ref, "resource_id", None)
        path = getattr(ref, "path", None)
        if isinstance(record_id, str) and record_id:
            item["ref"] = {"record": record_id}
            if record_producers and record_id in record_producers:
                item["source"] = deepcopy(record_producers[record_id])
        elif isinstance(resource_id, str) and resource_id:
            item["ref"] = {"resource": resource_id}
        elif path is not None:
            item["ref"] = {"file": str(path)}
        else:
            item["ref"] = {"display": _binding_display(ref)}
        declared_port = (declared_ports or {}).get(label)
        if declared_port is not None:
            item["kind"] = declared_port.kind
            item["declared"] = declared_port.render()
            if declared_port.contract is not None:
                item["contract"] = declared_port.contract
            if declared_port.optional:
                item["optional"] = True
        payload.append(item)
    return payload


def pipeline_writes_payload(step, *, runtime: ReaderRuntime) -> list[dict[str, object]]:
    plugin_cls = runtime.plugins.resolve_descriptor(step.plugin).cls
    outputs: list[dict[str, object]] = []
    for output_name, port in plugin_cls.output_ports().items():
        if port.kind == "dataframe":
            record_ref = (step.writes or {}).get(output_name, OutputRef(record_id=f"{step.id}/{output_name}"))
            output: dict[str, object] = {
                "label": output_name,
                "kind": port.kind,
                "declared": port.render(),
                "display": record_ref.record_id,
                "contract": port.contract,
                "ref": output_ref_to_dict(record_ref),
            }
            surface = _output_surface_payload(port)
            if surface is not None:
                output["surface"] = surface
            outputs.append(output)
            continue
        outputs.append(
            {
                "label": output_name,
                "kind": port.kind,
                "declared": port.render(),
                "display": output_name,
            }
        )
    return outputs


def binding_display(ref) -> str:
    return _binding_display(ref)


def _binding_display(ref) -> str:
    record_id = getattr(ref, "record_id", None)
    if isinstance(record_id, str) and record_id:
        return record_id
    resource_id = getattr(ref, "resource_id", None)
    if isinstance(resource_id, str) and resource_id:
        return f"resource({resource_id})"
    path = getattr(ref, "path", None)
    if path is not None:
        return str(path)
    return input_ref_display(ref)


def _output_surface_payload(port) -> dict[str, object] | None:
    surface = getattr(port, "contract_surface", None)
    if surface is None:
        return None
    return {
        "minimum": surface.minimum,
        "runtime_mode": surface.runtime_mode,
        "promoted": list(surface.promoted),
        "note": surface.note,
        "rendered": surface.render(),
    }
