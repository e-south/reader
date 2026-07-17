from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml
from rich import box
from rich.table import Table

from reader.protocols import ProtocolBinding
from reader.protocols.model import ProtocolAnalysisChoiceRef, ProtocolBindingValueRef

from .runtime import compiled_workbench_payload, record_producer_map
from .semantics import semantic_program_payload


def _table(title: str) -> Table:
    return Table(
        title=f"[title]{title}[/title]",
        title_justify="left",
        header_style="bold",
        box=box.ROUNDED,
        expand=True,
        show_lines=False,
        show_edge=True,
    )


def protocol_surface_rows(fields) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for field in fields:
        rows.extend(field.iter_rows())
    return rows


def _protocol_field_payload(field, *, prefix: str = "") -> dict[str, object]:
    path = f"{prefix}{field.key}"
    payload: dict[str, object] = {
        "key": field.key,
        "path": path,
        "kind": field.kind,
        "summary": field.summary,
        "required": field.required,
        "allow_none": field.allow_none,
        "allow_unknown": field.allow_unknown,
    }
    if field.has_default:
        payload["default"] = deepcopy(field.default)
    if field.choices:
        payload["choices"] = list(field.choices)
    if field.children:
        payload["children"] = [_protocol_field_payload(child, prefix=f"{path}.") for child in field.children]
    return payload


def protocol_surface_payload(fields) -> list[dict[str, object]]:
    return [_protocol_field_payload(field) for field in fields]


def protocol_surface_table(title: str, rows: list[tuple[str, str, str, str, str]]) -> Table:
    table = _table(title)
    table.add_column("Path", style="accent")
    table.add_column("Type")
    table.add_column("Required", justify="center")
    table.add_column("Default")
    table.add_column("Summary")
    for path, kind, required, default, summary in rows:
        table.add_row(path, kind, required, default, summary)
    return table


def protocol_plot_profiles_table(descriptor) -> Table:
    table = _table("Plot Profiles")
    table.add_column("id", style="accent")
    table.add_column("figures")
    table.add_column("summary")
    for item in descriptor.plot_profiles:
        table.add_row(item.id, ", ".join(item.figures), item.summary)
    return table


def protocol_plot_outputs_table(descriptor) -> Table:
    table = _table("Plot Outputs")
    table.add_column("id", style="accent")
    table.add_column("kind")
    table.add_column("primary", justify="center")
    table.add_column("summary")
    for item in descriptor.figures:
        table.add_row(item.id, item.kind, "yes" if item.primary else "no", item.summary)
    return table


def protocol_artifacts_table(descriptor) -> Table:
    table = _table("Export Artifacts")
    table.add_column("id", style="accent")
    table.add_column("summary")
    for item in descriptor.artifacts:
        table.add_row(item.id, item.summary)
    return table


def protocol_pipeline_table(steps) -> Table:
    table = _table("Default Pipeline")
    table.add_column("#", justify="right", style="muted")
    table.add_column("stage", style="accent")
    table.add_column("id", overflow="fold")
    table.add_column("plugin", overflow="fold")
    for idx, step in enumerate(steps, 1):
        stage = step.plugin.split("/", 1)[0]
        table.add_row(str(idx), stage, step.id, step.plugin)
    return table


def protocol_surface_impl_table(title: str, steps, summaries: dict[str, str], *, binding_display) -> Table:
    table = _table(title)
    table.add_column("id", style="accent", overflow="fold")
    table.add_column("plugin", overflow="fold")
    table.add_column("from", overflow="fold")
    table.add_column("summary", overflow="fold")
    for step in steps:
        from_refs = ", ".join(f"{label} <- {binding_display(ref)}" for label, ref in (step.reads or {}).items()) or "—"
        table.add_row(step.id, step.plugin, from_refs, summaries.get(step.id, "—"))
    return table


_EXAMPLE_OMIT = object()


def _protocol_field_example_value(field) -> object:
    if field.kind == "mapping":
        example: dict[str, object] = {}
        for child in field.children:
            child_value = _protocol_field_example_value(child)
            if child_value is _EXAMPLE_OMIT:
                continue
            example[child.key] = child_value
        if example:
            return example
        if field.has_default:
            return deepcopy(field.default)
        if field.required:
            return {}
        return _EXAMPLE_OMIT
    if field.has_default:
        return deepcopy(field.default)
    if field.required:
        return "<required>"
    return _EXAMPLE_OMIT


def _protocol_surface_example(fields) -> dict[str, object]:
    example: dict[str, object] = {}
    for field in fields:
        value = _protocol_field_example_value(field)
        if value is _EXAMPLE_OMIT:
            continue
        example[field.key] = value
    return example


def protocol_authoring_output_payload(descriptor) -> dict[str, object]:
    return {
        "notebook_policy": {
            "default_template": descriptor.execution.notebook.default_template,
            "allowed_templates": list(descriptor.execution.notebook.allowed_templates),
            "summary": descriptor.execution.notebook.summary,
        },
        "default_plot_profile": descriptor.default_plot_profile,
        "plot_profiles": [
            {
                "id": item.id,
                "figures": list(item.figures),
                "summary": item.summary,
            }
            for item in descriptor.plot_profiles
        ],
        "figures": [
            {
                "id": item.id,
                "kind": item.kind,
                "primary": item.primary,
                "summary": item.summary,
            }
            for item in descriptor.figures
        ],
        "artifacts": [
            {
                "id": item.id,
                "summary": item.summary,
                "default": item.default,
            }
            for item in descriptor.artifacts
        ],
    }


def protocol_example_document(descriptor) -> dict[str, object]:
    protocol_block: dict[str, object] = {"id": descriptor.protocol}
    inputs = _protocol_surface_example(descriptor.input_fields)
    if inputs:
        protocol_block["inputs"] = inputs
    analysis = _protocol_surface_example(descriptor.analysis_fields)
    if analysis:
        protocol_block["analysis"] = analysis

    outputs: dict[str, object] = {
        "notebook": {"template": descriptor.execution.notebook.default_template},
    }
    if descriptor.plot_profiles or descriptor.figures:
        plots: dict[str, object] = {}
        if descriptor.default_plot_profile is not None:
            plots["profile"] = descriptor.default_plot_profile
        elif descriptor.figures:
            plots["include"] = [item.id for item in descriptor.figures if item.primary]
        if plots:
            outputs["plots"] = plots
    default_artifacts = [item.id for item in descriptor.artifacts if item.default]
    if default_artifacts:
        outputs["exports"] = {"include": default_artifacts}
    protocol_block["outputs"] = outputs

    return {
        "schema": "reader/v8",
        "experiment": {"id": "example_experiment"},
        "protocol": protocol_block,
        "resources": {},
        "annotations": {},
    }


def protocol_example_config(descriptor) -> str:
    return yaml.safe_dump(protocol_example_document(descriptor), sort_keys=False)


def _runtime_default_value_payload(value: Any) -> Any:
    if isinstance(value, ProtocolBindingValueRef):
        payload: dict[str, object] = {
            "source": f"protocol.inputs.{value.key}",
        }
        if value.has_default:
            payload["default"] = _runtime_default_value_payload(value.default)
        else:
            payload["required"] = True
        return payload
    if isinstance(value, ProtocolAnalysisChoiceRef):
        payload: dict[str, object] = {
            "source": f"protocol.analysis.{value.key}",
            "cases": {key: _runtime_default_value_payload(item) for key, item in value.cases.items()},
        }
        if value.has_default:
            payload["default"] = _runtime_default_value_payload(value.default)
        else:
            payload["required"] = True
        return payload
    if isinstance(value, dict):
        return {key: _runtime_default_value_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_runtime_default_value_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_runtime_default_value_payload(item) for item in value]
    return deepcopy(value)


def protocol_runtime_defaults_payload(plugin_defaults) -> list[dict[str, object]]:
    return [
        {
            "plugin": item.plugin,
            "summary": item.summary,
            "parameters": _runtime_default_value_payload(item.with_),
        }
        for item in plugin_defaults
    ]


def protocol_descriptor_payload(descriptor, *, runtime) -> dict[str, object]:
    bound_protocol = runtime.bind_protocol(ProtocolBinding(id=descriptor.protocol))
    compiled_plan = bound_protocol.compile()
    semantic_program = compiled_plan.semantic_program
    record_producers = record_producer_map(compiled_plan.pipeline, runtime=runtime)
    compiled_payload = compiled_workbench_payload(
        bound_protocol=bound_protocol,
        pipeline_steps=compiled_plan.pipeline,
        plot_steps=compiled_plan.plots,
        export_steps=compiled_plan.exports,
        notebook_steps=compiled_plan.notebooks,
        runtime=runtime,
        record_producers=record_producers,
    )
    compiled_payload["semantic_program"] = semantic_program_payload(semantic_program)
    return {
        "protocol": descriptor.protocol,
        "domain": descriptor.domain,
        "family": descriptor.family,
        "summary": descriptor.summary,
        "tags": list(descriptor.tags),
        "authoring": {
            "inputs": protocol_surface_payload(descriptor.input_fields),
            "analysis": protocol_surface_payload(descriptor.analysis_fields),
            "outputs": protocol_authoring_output_payload(descriptor),
            "starter_config": protocol_example_document(descriptor),
        },
        "semantics": {
            "factors": [
                {
                    "name": item.name,
                    "role": item.role,
                    "summary": item.summary,
                    "required": item.required,
                    "repeatable": item.repeatable,
                }
                for item in descriptor.factors
            ],
            "effect_signs": [
                {
                    "target": item.target,
                    "expected_sign": item.expected_sign,
                    "summary": item.summary,
                }
                for item in descriptor.effect_signs
            ],
            "program": semantic_program_payload(semantic_program, include_execution=False),
        },
        "implementation": {
            "defaults": protocol_runtime_defaults_payload(descriptor.execution.plugin_defaults),
            "compiled": compiled_payload,
        },
    }
