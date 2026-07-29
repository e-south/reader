from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

from rich.console import Console
from rich.theme import Theme

from reader.errors import ConfigError
from reader.protocols import ProtocolBinding
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import build_workbench_decl
from reader.workbench.engine import normalize_log_level, run_spec, validation_summary
from reader.workbench.graph import resolve_workbench, select_workbench_specs
from reader.workbench.inspection.catalogs import workbench_surface_specs_payload
from reader.workbench.inspection.experiments import (
    experiment_explain_payload,
    experiment_identity_payload,
    experiment_inspect_payload,
)
from reader.workbench.inspection.results import record_catalog_payload
from reader.workbench.records import verify_record_store

from .models import (
    Experiment,
    ExperimentEvidence,
    ExperimentIdentity,
    InspectionResult,
    PlanResult,
    PluginCatalogResult,
    PluginDescriptorResult,
    PluginPort,
    PluginSummary,
    RecordCatalogResult,
    RecordRevision,
    RunResult,
    SelectedSteps,
    SurfaceCatalogResult,
    ValidationResult,
    VerificationResult,
)

_SILENT_CONSOLE_THEME = Theme(
    {
        "title": "bold cyan",
        "accent": "cyan",
        "ok": "bold green",
        "warn": "bold yellow",
        "error": "bold red",
        "muted": "dim",
        "path": "magenta",
    }
)


def open_experiment(path: str | Path, *, runtime: ReaderRuntime | None = None) -> Experiment:
    """Load and compile one experiment without creating experiment outputs."""

    config_path = Path(path).expanduser().resolve()
    if config_path.is_dir():
        config_path = config_path / "config.yaml"
    if not config_path.exists() or not config_path.is_file():
        raise ConfigError(f"Experiment config not found: {config_path}")
    active_runtime = runtime or builtin_runtime()
    spec = ReaderSpec.load(config_path)
    declaration = build_workbench_decl(spec, source_path=config_path, protocols=active_runtime.protocols)
    return Experiment(
        config_path=config_path,
        spec=spec,
        declaration=declaration,
        runtime=active_runtime,
    )


def inspect(experiment: Experiment) -> InspectionResult:
    payload = experiment_inspect_payload(
        job_path=experiment.config_path,
        spec=experiment.spec,
        decl=experiment.declaration,
        runtime=experiment.runtime,
    )
    return InspectionResult(
        experiment=_identity(payload["experiment"]),
        authoring=deepcopy(payload["authoring"]),
        semantics=deepcopy(payload["semantics"]),
        implementation=deepcopy(payload["implementation"]),
    )


def validate(experiment: Experiment, *, check_files: bool = True) -> ValidationResult:
    summary = validation_summary(
        experiment.declaration,
        check_files=check_files,
        exp_root=experiment.declaration.experiment.root,
        runtime=experiment.runtime,
    )
    summary_copy = deepcopy(summary)
    status = str(summary_copy.pop("status"))
    summary_fields = {
        "checks": summary_copy.pop("checks"),
        "counts": summary_copy.pop("counts"),
    }
    return ValidationResult(
        experiment=experiment.identity,
        check_files=check_files,
        status=status,
        summary=summary_fields,
        validation=summary_copy,
    )


def plan(experiment: Experiment) -> PlanResult:
    payload = experiment_explain_payload(
        job_path=experiment.config_path,
        spec=experiment.spec,
        decl=experiment.declaration,
        runtime=experiment.runtime,
    )
    implementation = payload["implementation"]
    return PlanResult(
        experiment=_identity(payload["experiment"]),
        plan=deepcopy(implementation["plan"]),
        compiled=deepcopy(implementation["compiled"]),
        semantics=deepcopy(payload["semantics"]),
    )


def plots(
    experiment: Experiment,
    *,
    only: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> SurfaceCatalogResult:
    only_ids = list(only)
    exclude_ids = list(exclude)
    workbench = resolve_workbench(experiment.declaration)
    selected = select_workbench_specs(
        list(workbench.plots),
        only=only_ids,
        exclude=exclude_ids,
        kind_label="plot",
    )
    bound_protocol = experiment.runtime.bind_protocol(experiment.declaration.experiment_semantics.protocol)
    payload = workbench_surface_specs_payload(
        job_path=experiment.config_path,
        decl=experiment.declaration,
        runtime=experiment.runtime,
        bound_protocol=bound_protocol,
        selected=selected,
        kind="plot",
        only=only_ids,
        exclude=exclude_ids,
    )
    return SurfaceCatalogResult(
        experiment=_identity(payload["experiment"]),
        kind="plot",
        protocol=str(payload["catalog"]["protocol"]),
        selection=deepcopy(payload["selection"]),
        summary=deepcopy(payload["summary"]),
        entries=tuple(deepcopy(payload["plots"])),
    )


def records(experiment: Experiment, *, include_history: bool = False) -> RecordCatalogResult:
    decl = experiment.declaration
    layout = decl.experiment_semantics.layout
    outputs_dir = layout.outputs_dir
    store = experiment.runtime.record_store(
        outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=decl.experiment.root,
        create=False,
    )
    if not store.catalog_exists():
        return RecordCatalogResult(
            experiment=experiment.identity,
            catalog_exists=False,
            catalog={"path": str(store.records_path), "outputs_root": str(outputs_dir)},
            include_history=include_history,
            summary={
                "records": 0,
                "history": {"included": include_history, "revisions": 0 if include_history else None},
                "by_kind": {},
                "by_producer": {},
            },
            entries=(),
        )
    payload = record_catalog_payload(
        experiment=experiment_identity_payload(job_path=experiment.config_path, decl=decl),
        store=store,
        outputs_dir=outputs_dir,
        runtime=experiment.runtime,
        include_history=include_history,
    )
    return RecordCatalogResult(
        experiment=_identity(payload["experiment"]),
        catalog_exists=True,
        catalog=deepcopy(payload["catalog"]),
        include_history=include_history,
        summary=deepcopy(payload["summary"]),
        entries=tuple(deepcopy(payload["records"])),
    )


def verify(experiment: Experiment) -> VerificationResult:
    """Verify the current catalog without creating or changing experiment outputs."""
    decl = experiment.declaration
    layout = decl.experiment_semantics.layout
    store = experiment.runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=decl.experiment.root,
        create=False,
    )
    report = verify_record_store(
        store,
        experiment_root=decl.experiment.root,
        expected_config_digest=decl.config_digest,
    )
    return VerificationResult(
        experiment=experiment.identity,
        status=str(report["status"]),
        summary=deepcopy(report["summary"]),
        issues=tuple(deepcopy(report["issues"])),
        records=tuple(deepcopy(report["records"])),
    )


def run(
    experiment: Experiment,
    *,
    from_step: str | None = None,
    until_step: str | None = None,
    only: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> RunResult:
    """Run an experiment pipeline through the same execution path as the CLI."""

    from_step = _step_selector(from_step, field="from_step")
    until_step = _step_selector(until_step, field="until_step")
    only = _step_selector(only, field="only")
    if only is not None and (from_step is not None or until_step is not None):
        raise ConfigError("only cannot be combined with from_step or until_step")
    normalize_log_level(log_level)
    if not dry_run and experiment.declaration.experiment.lifecycle != "active":
        lifecycle = experiment.declaration.experiment.lifecycle
        raise ConfigError(f"Experiment lifecycle {lifecycle!r} is not runnable")

    execution = run_spec(
        experiment.declaration,
        resume_from=only if only is not None else from_step,
        until=only if only is not None else until_step,
        dry_run=dry_run,
        log_level=log_level,
        verbose=False,
        console=Console(quiet=True, theme=_SILENT_CONSOLE_THEME),
        include_pipeline=True,
        include_plots=False,
        include_exports=False,
        show_next_steps=False,
        runtime=experiment.runtime,
    )
    return RunResult(
        experiment=experiment.identity,
        invocation_id=execution.invocation_id,
        operation=execution.operation,
        status=execution.status,
        dry_run=execution.dry_run,
        selected_steps=SelectedSteps(
            pipeline=execution.selected_steps.pipeline,
            plots=execution.selected_steps.plots,
            exports=execution.selected_steps.exports,
        ),
        produced_record_revisions=tuple(
            RecordRevision(
                record_id=revision.record_id,
                revision=revision.revision,
                revision_digest=revision.revision_digest,
            )
            for revision in execution.produced_record_revisions
        ),
        ledger_path=str(execution.ledger_path) if execution.ledger_path is not None else None,
    )


def _step_selector(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty step id")
    return value.strip()


def plugins(
    *,
    category: str | None = None,
    domain: str | None = None,
    family: str | None = None,
    protocol: str | None = None,
    runtime: ReaderRuntime | None = None,
) -> PluginCatalogResult:
    active_runtime = runtime or builtin_runtime()
    descriptors = list(active_runtime.plugins.catalog().filter(category=category, domain=domain, family=family))
    if protocol is not None:
        bound_protocol = active_runtime.bind_protocol(ProtocolBinding(id=protocol))
        compiled = bound_protocol.compile()
        allowed = {step.plugin for step in (*compiled.pipeline, *compiled.plots, *compiled.exports)}
        descriptors = [descriptor for descriptor in descriptors if descriptor.plugin in allowed]
    plugin_items = tuple(_plugin_summary(descriptor) for descriptor in descriptors)
    return PluginCatalogResult(
        selection={"category": category, "domain": domain, "family": family, "protocol": protocol},
        summary={
            "plugins": len(plugin_items),
            "by_category": _counts(item.category for item in plugin_items),
            "by_domain": _counts(item.domain for item in plugin_items),
            "by_family": _counts(item.family for item in plugin_items),
        },
        plugins=plugin_items,
    )


def describe_plugin(
    plugin_id: str,
    *,
    runtime: ReaderRuntime | None = None,
) -> PluginDescriptorResult:
    active_runtime = runtime or builtin_runtime()
    descriptor = active_runtime.plugins.resolve_descriptor(plugin_id)
    plugin_cls = descriptor.cls
    return PluginDescriptorResult(
        plugin=_plugin_summary(descriptor),
        config_schema=deepcopy(plugin_cls.ConfigModel.model_json_schema()),
        input_ports=tuple(
            PluginPort(
                name=port.name,
                kind=port.kind,
                optional=port.optional,
                contract=port.contract,
            )
            for _, port in sorted(plugin_cls.input_ports().items())
        ),
        output_ports=tuple(
            PluginPort(
                name=port.name,
                kind=port.kind,
                optional=False,
                contract=port.contract,
                contract_surface=_contract_surface(port),
            )
            for _, port in sorted(plugin_cls.output_ports().items())
        ),
    )


def _identity(payload: object) -> ExperimentIdentity:
    if not isinstance(payload, dict):
        raise TypeError("Experiment identity payload must be a mapping")
    evidence_payload = payload.get("evidence")
    evidence = None
    if evidence_payload is not None:
        if not isinstance(evidence_payload, dict):
            raise TypeError("Experiment evidence payload must be a mapping when present")
        evidence = ExperimentEvidence(
            data_class=str(evidence_payload["data_class"]),
            data_class_reason=str(evidence_payload["data_class_reason"]),
            replicate_kind=str(evidence_payload["replicate_kind"]),
            replicate_identity_field=(
                str(evidence_payload["replicate_identity_field"])
                if evidence_payload.get("replicate_identity_field") is not None
                else None
            ),
        )
    return ExperimentIdentity(
        id=str(payload["id"]),
        title=str(payload["title"]),
        lifecycle=str(payload["lifecycle"]),
        protocol=str(payload["protocol"]),
        config=str(payload["config"]),
        root=str(payload["root"]),
        evidence=evidence,
    )


def _plugin_summary(descriptor) -> PluginSummary:
    return PluginSummary(
        plugin=descriptor.plugin,
        key=descriptor.key,
        category=str(descriptor.category),
        domain=descriptor.domain,
        family=descriptor.family,
        summary=descriptor.summary,
        implementation=f"{descriptor.cls.__module__}.{descriptor.cls.__name__}",
    )


def _contract_surface(port) -> dict[str, object] | None:
    surface = port.contract_surface
    if surface is None:
        return None
    return {
        "minimum": surface.minimum,
        "runtime_mode": surface.runtime_mode,
        "promoted": list(surface.promoted),
        "note": surface.note,
        "rendered": surface.render(),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(value for value in values if value).items()))
