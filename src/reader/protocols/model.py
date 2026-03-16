from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from reader.domains.semantics import PluginDomain, validate_plugin_domain
from reader.errors import ConfigError
from reader.workbench.decl.model import NotebookTemplateCallDecl, PluginStepDecl

MetricStage = Literal["raw", "support", "derived", "comparison", "summary", "ranking", "qc", "burden", "leakiness"]
FigureKind = Literal["qc", "kinetics", "summary", "ranking", "architecture"]
RankingDirection = Literal["higher_is_better", "lower_is_better"]
DeliverableSurface = Literal["plots", "exports"]

_UNSET = object()


@dataclass(frozen=True)
class ProtocolBinding:
    id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    deliverables: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        protocol_id = str(self.id).strip()
        if not protocol_id:
            raise ValueError("ProtocolBinding.id must be a non-empty string.")
        object.__setattr__(self, "id", protocol_id)
        object.__setattr__(self, "parameters", dict(self.parameters or {}))
        object.__setattr__(self, "analysis", dict(self.analysis or {}))
        object.__setattr__(self, "deliverables", dict(self.deliverables or {}))


@dataclass(frozen=True)
class ProtocolBindingValueRef:
    key: str
    default: Any = _UNSET

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("ProtocolBindingValueRef.key must be a non-empty string.")
        object.__setattr__(self, "key", self.key.strip())

    @property
    def has_default(self) -> bool:
        return self.default is not _UNSET


def binding_value(key: str, default: Any = _UNSET) -> ProtocolBindingValueRef:
    return ProtocolBindingValueRef(key=key, default=default)


@dataclass(frozen=True)
class ProtocolFactorSpec:
    name: str
    role: str
    summary: str
    required: bool = True
    repeatable: bool = False

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("ProtocolFactorSpec.name must be a non-empty string.")
        if not str(self.role).strip():
            raise ValueError("ProtocolFactorSpec.role must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolFactorSpec.summary must be a non-empty string.")


@dataclass(frozen=True)
class ProtocolControlRule:
    id: str
    summary: str
    match_on: tuple[str, ...] = ()
    control_selector: str | None = None

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolControlRule.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolControlRule.summary must be a non-empty string.")
        object.__setattr__(self, "match_on", tuple(str(value) for value in self.match_on))
        if self.control_selector is not None and not str(self.control_selector).strip():
            raise ValueError("ProtocolControlRule.control_selector must be a non-empty string when provided.")


@dataclass(frozen=True)
class ProtocolWindowSpec:
    id: str
    summary: str
    anchor: str
    selector: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolWindowSpec.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolWindowSpec.summary must be a non-empty string.")
        if not str(self.anchor).strip():
            raise ValueError("ProtocolWindowSpec.anchor must be a non-empty string.")
        if not str(self.selector).strip():
            raise ValueError("ProtocolWindowSpec.selector must be a non-empty string.")
        object.__setattr__(self, "params", dict(self.params or {}))


@dataclass(frozen=True)
class ProtocolMetricSpec:
    id: str
    stage: MetricStage
    summary: str
    formula: str
    depends_on: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolMetricSpec.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolMetricSpec.summary must be a non-empty string.")
        if not str(self.formula).strip():
            raise ValueError("ProtocolMetricSpec.formula must be a non-empty string.")
        object.__setattr__(self, "depends_on", tuple(str(value) for value in self.depends_on))
        object.__setattr__(self, "notes", tuple(str(value) for value in self.notes))


@dataclass(frozen=True)
class ProtocolEffectSignSpec:
    target: str
    expected_sign: Literal["positive", "negative"]
    summary: str

    def __post_init__(self) -> None:
        if not str(self.target).strip():
            raise ValueError("ProtocolEffectSignSpec.target must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolEffectSignSpec.summary must be a non-empty string.")


@dataclass(frozen=True)
class ProtocolFigureSpec:
    id: str
    kind: FigureKind
    summary: str
    primary: bool = False

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolFigureSpec.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolFigureSpec.summary must be a non-empty string.")


@dataclass(frozen=True)
class ProtocolDeliverableSpec:
    id: str
    surface: DeliverableSurface
    summary: str
    default: bool = False

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolDeliverableSpec.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolDeliverableSpec.summary must be a non-empty string.")


@dataclass(frozen=True)
class ProtocolRankingSpec:
    primary_metric: str
    direction: RankingDirection
    penalties: tuple[str, ...] = ()
    supporting_metrics: tuple[str, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        if not str(self.primary_metric).strip():
            raise ValueError("ProtocolRankingSpec.primary_metric must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolRankingSpec.summary must be a non-empty string.")
        object.__setattr__(self, "penalties", tuple(str(value) for value in self.penalties))
        object.__setattr__(self, "supporting_metrics", tuple(str(value) for value in self.supporting_metrics))


@dataclass(frozen=True)
class ProtocolPluginDefaultsSpec:
    plugin: str
    summary: str
    with_: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plugin, str) or not self.plugin.strip():
            raise ValueError("ProtocolPluginDefaultsSpec.plugin must be a non-empty string.")
        if "/" not in self.plugin:
            raise ValueError("ProtocolPluginDefaultsSpec.plugin must be 'category/key'.")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("ProtocolPluginDefaultsSpec.summary must be a non-empty string.")
        object.__setattr__(self, "plugin", self.plugin.strip())
        object.__setattr__(self, "with_", dict(self.with_ or {}))


@dataclass(frozen=True)
class ProtocolNotebookPolicy:
    default_template: str
    allowed_templates: tuple[str, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        default_template = str(self.default_template).strip()
        if not default_template:
            raise ValueError("ProtocolNotebookPolicy.default_template must be a non-empty string.")
        summary = str(self.summary).strip()
        if not summary:
            raise ValueError("ProtocolNotebookPolicy.summary must be a non-empty string.")
        allowed = tuple(str(value).strip() for value in self.allowed_templates if str(value).strip())
        if not allowed:
            allowed = (default_template,)
        elif default_template not in allowed:
            allowed = (default_template, *allowed)
        object.__setattr__(self, "default_template", default_template)
        object.__setattr__(self, "allowed_templates", allowed)
        object.__setattr__(self, "summary", summary)


@dataclass(frozen=True)
class CompiledProtocolPlan:
    runtime: dict[str, Any] = field(default_factory=dict)
    pipeline: tuple[PluginStepDecl, ...] = ()
    plots: tuple[PluginStepDecl, ...] = ()
    exports: tuple[PluginStepDecl, ...] = ()
    notebooks: tuple[NotebookTemplateCallDecl, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", dict(self.runtime or {}))
        object.__setattr__(self, "pipeline", tuple(self.pipeline or ()))
        object.__setattr__(self, "plots", tuple(self.plots or ()))
        object.__setattr__(self, "exports", tuple(self.exports or ()))
        object.__setattr__(self, "notebooks", tuple(self.notebooks or ()))


ProtocolCompiler = Callable[["BoundProtocol"], CompiledProtocolPlan]


@dataclass(frozen=True)
class ProtocolExecutionPlan:
    notebook: ProtocolNotebookPolicy
    plugin_defaults: tuple[ProtocolPluginDefaultsSpec, ...] = ()
    compiler: ProtocolCompiler | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for item in self.plugin_defaults:
            if item.plugin in seen:
                raise ValueError(f"Duplicate protocol plugin defaults for {item.plugin!r}.")
            seen.add(item.plugin)
        if self.compiler is not None and not callable(self.compiler):
            raise ValueError("ProtocolExecutionPlan.compiler must be callable when provided.")


@dataclass(frozen=True)
class ProtocolDescriptor:
    protocol: str
    domain: PluginDomain
    family: str
    summary: str
    execution: ProtocolExecutionPlan
    tags: tuple[str, ...] = ()
    factors: tuple[ProtocolFactorSpec, ...] = ()
    control_rules: tuple[ProtocolControlRule, ...] = ()
    windows: tuple[ProtocolWindowSpec, ...] = ()
    metrics: tuple[ProtocolMetricSpec, ...] = ()
    effect_signs: tuple[ProtocolEffectSignSpec, ...] = ()
    figures: tuple[ProtocolFigureSpec, ...] = ()
    deliverables: tuple[ProtocolDeliverableSpec, ...] = ()
    ranking: ProtocolRankingSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", validate_plugin_domain(self.domain))
        if not str(self.protocol).strip():
            raise ValueError("ProtocolDescriptor.protocol must be a non-empty string.")
        if not str(self.family).strip():
            raise ValueError("ProtocolDescriptor.family must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolDescriptor.summary must be a non-empty string.")
        object.__setattr__(self, "tags", tuple(str(value) for value in self.tags))
        seen: set[tuple[str, str]] = set()
        for item in self.deliverables:
            key = (item.surface, item.id)
            if key in seen:
                raise ValueError(f"Duplicate protocol deliverable {item.surface}:{item.id!r}.")
            seen.add(key)


@dataclass(frozen=True)
class BoundProtocol:
    descriptor: ProtocolDescriptor
    parameters: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    deliverables: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters or {}))
        object.__setattr__(self, "analysis", dict(self.analysis or {}))
        object.__setattr__(self, "deliverables", dict(self.deliverables or {}))

    @property
    def id(self) -> str:
        return self.descriptor.protocol

    @property
    def domain(self) -> PluginDomain:
        return self.descriptor.domain

    @property
    def family(self) -> str:
        return self.descriptor.family

    @property
    def summary(self) -> str:
        return self.descriptor.summary

    @property
    def execution(self) -> ProtocolExecutionPlan:
        return self.descriptor.execution

    @property
    def default_notebook_template(self) -> str:
        return self.execution.notebook.default_template

    @property
    def allowed_notebook_templates(self) -> tuple[str, ...]:
        return self.execution.notebook.allowed_templates

    def allows_notebook_template(self, template: str) -> bool:
        return template in self.allowed_notebook_templates

    def resolve_notebook_template(
        self,
        *,
        explicit_template: str | None = None,
        configured_template: str | None = None,
    ) -> str:
        selected = explicit_template or configured_template or self.default_notebook_template
        if not self.allows_notebook_template(selected):
            options = ", ".join(self.allowed_notebook_templates) or "—"
            raise ConfigError(
                f"Protocol {self.id!r} does not allow notebook template {selected!r}. Allowed templates: {options}"
            )
        return selected

    def compile(self) -> CompiledProtocolPlan:
        compiler = self.execution.compiler
        if compiler is None:
            raise ConfigError(f"Protocol {self.id!r} does not define an executable compiler.")
        plan = compiler(self)
        if not isinstance(plan, CompiledProtocolPlan):
            raise ConfigError(
                f"Protocol {self.id!r} compiler returned {type(plan).__name__}, expected CompiledProtocolPlan."
            )
        notebooks = plan.notebooks
        if not notebooks:
            selected_template = self.resolve_notebook_template(configured_template=self.configured_notebook_template())
            notebooks = (NotebookTemplateCallDecl(id="default", template=selected_template),)
        for entry in notebooks:
            self.resolve_notebook_template(explicit_template=entry.template)
        return CompiledProtocolPlan(
            runtime=plan.runtime,
            pipeline=plan.pipeline,
            plots=plan.plots,
            exports=plan.exports,
            notebooks=notebooks,
        )

    def effective_plugin_config(self, *, plugin_id: str, step_with: dict[str, Any] | None = None) -> dict[str, Any]:
        defaults = self._protocol_plugin_defaults(plugin_id)
        return _deep_merge(defaults, dict(step_with or {}))

    def configured_notebook_template(self) -> str | None:
        block = self._deliverable_block("notebook")
        template = block.get("template")
        if template is None:
            return None
        if not isinstance(template, str) or not template.strip():
            raise ConfigError(f"protocol.deliverables.notebook.template for {self.id!r} must be a non-empty string")
        return template.strip()

    def select_deliverables(
        self,
        *,
        surface: DeliverableSurface,
        defaults: tuple[str, ...],
        allowed: set[str],
    ) -> tuple[str, ...]:
        block = self._deliverable_block(surface)
        include = self._validate_deliverable_ids(
            block.get("include", ()),
            where=f"protocol.deliverables.{surface}.include",
            allowed=allowed,
        )
        exclude = set(
            self._validate_deliverable_ids(
                block.get("exclude", ()),
                where=f"protocol.deliverables.{surface}.exclude",
                allowed=allowed,
            )
        )
        for deliverable_id in self._validate_deliverable_settings(surface=surface, allowed=allowed):
            if deliverable_id not in allowed:
                options = ", ".join(sorted(allowed)) or "—"
                raise ConfigError(
                    f"protocol.deliverables.{surface}.settings.{deliverable_id!r} is unknown for {self.id!r}. "
                    f"Available ids: {options}"
                )
        selected: list[str] = []
        for deliverable_id in (*defaults, *include):
            if deliverable_id in exclude:
                continue
            if deliverable_id not in selected:
                selected.append(deliverable_id)
        return tuple(selected)

    def deliverable_settings(self, *, surface: DeliverableSurface, deliverable_id: str) -> dict[str, Any]:
        block = self._deliverable_block(surface)
        settings = block.get("settings", {})
        if not isinstance(settings, dict):
            raise ConfigError(f"protocol.deliverables.{surface}.settings for {self.id!r} must be a mapping")
        configured = settings.get(deliverable_id, {})
        if configured is None:
            return {}
        if not isinstance(configured, dict):
            raise ConfigError(
                f"protocol.deliverables.{surface}.settings.{deliverable_id!r} for {self.id!r} must be a mapping"
            )
        return deepcopy(configured)

    def _protocol_plugin_defaults(self, plugin_id: str) -> dict[str, Any]:
        for item in self.execution.plugin_defaults:
            if item.plugin == plugin_id:
                return self._resolve_binding_refs(item.with_, where=f"protocol {self.id} plugin {plugin_id}")
        return {}

    def _deliverable_block(self, surface: str) -> dict[str, Any]:
        raw = self.deliverables.get(surface, {})
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ConfigError(f"protocol.deliverables.{surface} for {self.id!r} must be a mapping")
        return dict(raw)

    def _validate_deliverable_ids(self, raw: Any, *, where: str, allowed: set[str]) -> tuple[str, ...]:
        if raw in (None, ()):
            return ()
        if not isinstance(raw, list):
            raise ConfigError(f"{where} for {self.id!r} must be a list of deliverable ids")
        values: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise ConfigError(f"{where} for {self.id!r} must contain only non-empty deliverable ids")
            deliverable_id = item.strip()
            if deliverable_id not in allowed:
                options = ", ".join(sorted(allowed)) or "—"
                raise ConfigError(f"{where} contains unknown deliverable {deliverable_id!r}. Available ids: {options}")
            values.append(deliverable_id)
        return tuple(values)

    def _validate_deliverable_settings(self, *, surface: str, allowed: set[str]) -> tuple[str, ...]:
        block = self._deliverable_block(surface)
        settings = block.get("settings", {})
        if settings in (None, {}):
            return ()
        if not isinstance(settings, dict):
            raise ConfigError(f"protocol.deliverables.{surface}.settings for {self.id!r} must be a mapping")
        ids: list[str] = []
        for deliverable_id, config in settings.items():
            if not isinstance(deliverable_id, str) or not deliverable_id.strip():
                raise ConfigError(
                    f"protocol.deliverables.{surface}.settings for {self.id!r} must use non-empty deliverable ids"
                )
            if not isinstance(config, dict):
                raise ConfigError(
                    f"protocol.deliverables.{surface}.settings.{deliverable_id!r} for {self.id!r} must be a mapping"
                )
            ids.append(deliverable_id.strip())
        return tuple(ids)

    def _resolve_binding_refs(self, value: Any, *, where: str) -> Any:
        if isinstance(value, ProtocolBindingValueRef):
            found, resolved = self._lookup_parameter_value(value.key)
            if found:
                return deepcopy(resolved)
            if value.has_default:
                return deepcopy(value.default)
            raise ConfigError(f"{where} requires protocol.parameters.{value.key}")
        if isinstance(value, dict):
            return {key: self._resolve_binding_refs(item, where=where) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_binding_refs(item, where=where) for item in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_binding_refs(item, where=where) for item in value)
        return deepcopy(value)

    def _lookup_parameter_value(self, key: str) -> tuple[bool, Any]:
        current: Any = self.parameters
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current


def _deep_merge(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = _deep_merge(merged[key], value)
                continue
            merged[key] = deepcopy(value)
    return merged


class ProtocolCatalog:
    def __init__(self, descriptors: list[ProtocolDescriptor]):
        self._descriptors = tuple(sorted(descriptors, key=lambda item: (item.domain, item.family, item.protocol)))
        by_id: dict[str, ProtocolDescriptor] = {}
        for item in self._descriptors:
            if item.protocol in by_id:
                raise ConfigError(f"Duplicate protocol {item.protocol!r}.")
            by_id[item.protocol] = item
        self._by_id = by_id

    def all(self) -> tuple[ProtocolDescriptor, ...]:
        return self._descriptors

    def resolve(self, protocol_id: str) -> ProtocolDescriptor:
        try:
            return self._by_id[protocol_id]
        except KeyError:
            options = ", ".join(sorted(item.protocol for item in self._descriptors)) or "—"
            raise ConfigError(f"Unknown protocol {protocol_id!r}. Available protocols: {options}") from None

    def bind(self, binding: ProtocolBinding) -> BoundProtocol:
        return BoundProtocol(
            descriptor=self.resolve(binding.id),
            parameters=binding.parameters,
            analysis=binding.analysis,
            deliverables=binding.deliverables,
        )

    def list(self, *, domain: str | None = None, family: str | None = None) -> list[tuple[str, str]]:
        return [
            (item.protocol, item.summary)
            for item in self._descriptors
            if (domain is None or item.domain == domain) and (family is None or item.family == family)
        ]
