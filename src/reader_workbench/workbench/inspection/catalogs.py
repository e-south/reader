from __future__ import annotations

from collections import Counter
from copy import deepcopy

from reader_workbench.workbench.graph import resolve_workbench

from .experiments import experiment_identity_payload
from .runtime import export_output_summaries, plot_output_summaries, record_producer_map, spec_step_payload


def _sorted_counter(values: list[str]) -> dict[str, int]:
    counts = Counter(value for value in values if value)
    return dict(sorted(counts.items()))


def workbench_surface_catalog_payload(
    *,
    experiment: dict[str, object],
    protocol: str,
    kind: str,
    only: list[str],
    exclude: list[str],
    entries: list[dict[str, object]],
) -> dict[str, object]:
    payload_key = f"{kind}s"
    summary_key = payload_key
    plugin_ids = [str(entry.get("plugin") or "") for entry in entries]
    domains = [
        str(semantics.get("domain") or "")
        for semantics in (entry.get("semantics") for entry in entries)
        if isinstance(semantics, dict)
    ]
    families = [
        str(semantics.get("family") or "")
        for semantics in (entry.get("semantics") for entry in entries)
        if isinstance(semantics, dict)
    ]
    return {
        "experiment": deepcopy(experiment),
        "catalog": {
            "kind": kind,
            "protocol": protocol,
        },
        "selection": {
            "only": list(only),
            "exclude": list(exclude),
        },
        "summary": {
            summary_key: len(entries),
            "by_plugin": _sorted_counter(plugin_ids),
            "by_domain": _sorted_counter(domains),
            "by_family": _sorted_counter(families),
        },
        payload_key: deepcopy(entries),
    }


def plugin_registry_payload(
    *,
    descriptors,
    category: str | None,
    domain: str | None,
    family: str | None,
    protocol: str | None,
) -> dict[str, object]:
    plugins = [
        {
            "category": descriptor.category,
            "domain": descriptor.domain,
            "family": descriptor.family,
            "key": descriptor.key,
            "plugin": descriptor.plugin,
            "summary": descriptor.summary,
            "class": f"{descriptor.cls.__module__}.{descriptor.cls.__name__}",
        }
        for descriptor in descriptors
    ]
    return {
        "selection": {
            "category": category,
            "domain": domain,
            "family": family,
            "protocol": protocol,
        },
        "summary": {
            "plugins": len(plugins),
            "by_category": _sorted_counter([str(item["category"]) for item in plugins]),
            "by_domain": _sorted_counter([str(item["domain"]) for item in plugins]),
            "by_family": _sorted_counter([str(item["family"]) for item in plugins]),
        },
        "plugins": plugins,
    }


def workbench_surface_specs_payload(
    *,
    job_path,
    decl,
    runtime,
    bound_protocol,
    selected,
    kind: str,
    only: list[str],
    exclude: list[str],
) -> dict[str, object]:
    workbench = resolve_workbench(decl)
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    summary_lookup = (
        plot_output_summaries(bound_protocol) if kind == "plot" else export_output_summaries(bound_protocol)
    )
    entries = [
        spec_step_payload(
            step,
            summary=summary_lookup.get(step.id, "—"),
            runtime=runtime,
            record_producers=record_producers,
        )
        for step in selected
    ]
    return workbench_surface_catalog_payload(
        experiment=experiment_identity_payload(job_path=job_path, decl=decl, protocol_id=bound_protocol.id),
        protocol=bound_protocol.id,
        kind=kind,
        only=only,
        exclude=exclude,
        entries=entries,
    )
