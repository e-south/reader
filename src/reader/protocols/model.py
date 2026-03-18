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
SemanticNodeKind = Literal["control_rule", "window", "metric", "ranking"]
SemanticExecutionStatus = Literal["compiled", "descriptive_only"]
ConfigFieldKind = Literal[
    "mapping",
    "string",
    "bool",
    "number",
    "integer",
    "string_list",
    "number_list",
    "scalar_list",
    "mapping_list",
    "scalar",
    "any",
]

_UNSET = object()


@dataclass(frozen=True)
class ProtocolSemanticProfileSpec:
    id: str
    family: str
    summary: str
    primary_metric: str | None = None
    primary_readout: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        profile_id = str(self.id).strip()
        family = str(self.family).strip()
        summary = str(self.summary).strip()
        if not profile_id:
            raise ValueError("ProtocolSemanticProfileSpec.id must be a non-empty string.")
        if not family:
            raise ValueError("ProtocolSemanticProfileSpec.family must be a non-empty string.")
        if not summary:
            raise ValueError("ProtocolSemanticProfileSpec.summary must be a non-empty string.")
        object.__setattr__(self, "id", profile_id)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "summary", summary)
        if self.primary_metric is not None and not str(self.primary_metric).strip():
            raise ValueError("ProtocolSemanticProfileSpec.primary_metric must be a non-empty string when provided.")
        if self.primary_readout is not None and not str(self.primary_readout).strip():
            raise ValueError("ProtocolSemanticProfileSpec.primary_readout must be a non-empty string when provided.")
        object.__setattr__(self, "tags", tuple(str(value).strip() for value in self.tags if str(value).strip()))


@dataclass(frozen=True)
class ProtocolSemanticProfileOverride:
    enabled: bool = True
    summary: str | None = None
    formula: str | None = None
    depends_on: tuple[str, ...] | None = None
    anchor: str | None = None
    selector: str | None = None
    params: dict[str, Any] | None = None
    match_on: tuple[str, ...] | None = None
    control_selector: str | None = None
    primary_metric: str | None = None
    direction: RankingDirection | None = None
    penalties: tuple[str, ...] | None = None
    supporting_metrics: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.summary is not None and not str(self.summary).strip():
            raise ValueError("ProtocolSemanticProfileOverride.summary must be a non-empty string when provided.")
        if self.formula is not None and not str(self.formula).strip():
            raise ValueError("ProtocolSemanticProfileOverride.formula must be a non-empty string when provided.")
        if self.anchor is not None and not str(self.anchor).strip():
            raise ValueError("ProtocolSemanticProfileOverride.anchor must be a non-empty string when provided.")
        if self.selector is not None and not str(self.selector).strip():
            raise ValueError("ProtocolSemanticProfileOverride.selector must be a non-empty string when provided.")
        if self.control_selector is not None and not str(self.control_selector).strip():
            raise ValueError(
                "ProtocolSemanticProfileOverride.control_selector must be a non-empty string when provided."
            )
        if self.primary_metric is not None and not str(self.primary_metric).strip():
            raise ValueError("ProtocolSemanticProfileOverride.primary_metric must be a non-empty string when provided.")
        if self.depends_on is not None:
            object.__setattr__(self, "depends_on", tuple(str(value) for value in self.depends_on if str(value).strip()))
        if self.params is not None:
            object.__setattr__(self, "params", dict(self.params or {}))
        if self.match_on is not None:
            object.__setattr__(self, "match_on", tuple(str(value) for value in self.match_on if str(value).strip()))
        if self.penalties is not None:
            object.__setattr__(self, "penalties", tuple(str(value) for value in self.penalties if str(value).strip()))
        if self.supporting_metrics is not None:
            object.__setattr__(
                self,
                "supporting_metrics",
                tuple(str(value) for value in self.supporting_metrics if str(value).strip()),
            )


@dataclass(frozen=True)
class ProtocolBinding:
    id: str
    inputs: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        protocol_id = str(self.id).strip()
        if not protocol_id:
            raise ValueError("ProtocolBinding.id must be a non-empty string.")
        object.__setattr__(self, "id", protocol_id)
        object.__setattr__(self, "inputs", dict(self.inputs or {}))
        object.__setattr__(self, "analysis", dict(self.analysis or {}))
        object.__setattr__(self, "outputs", dict(self.outputs or {}))


@dataclass(frozen=True)
class ProtocolConfigFieldSpec:
    key: str
    summary: str
    kind: ConfigFieldKind = "mapping"
    required: bool = False
    allow_none: bool = False
    choices: tuple[str, ...] = ()
    children: tuple[ProtocolConfigFieldSpec, ...] = ()
    allow_unknown: bool = False
    default: Any = _UNSET

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        if not key:
            raise ValueError("ProtocolConfigFieldSpec.key must be a non-empty string.")
        summary = str(self.summary).strip()
        if not summary:
            raise ValueError("ProtocolConfigFieldSpec.summary must be a non-empty string.")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "choices", tuple(str(value).strip() for value in self.choices if str(value).strip()))
        child_keys: set[str] = set()
        for child in self.children:
            if child.key in child_keys:
                raise ValueError(f"Duplicate child key {child.key!r} in ProtocolConfigFieldSpec {key!r}.")
            child_keys.add(child.key)
        if self.kind != "mapping" and self.children:
            raise ValueError("ProtocolConfigFieldSpec.children are only allowed when kind='mapping'.")

    @property
    def has_default(self) -> bool:
        return self.default is not _UNSET

    def render_default(self) -> str:
        if not self.has_default:
            return "—"
        if self.default is None:
            return "null"
        if isinstance(self.default, (str, int, float, bool)):
            return str(self.default)
        if isinstance(self.default, list):
            return "[" + ", ".join(str(item) for item in self.default) + "]"
        if isinstance(self.default, tuple):
            return "[" + ", ".join(str(item) for item in self.default) + "]"
        if isinstance(self.default, dict):
            keys = ", ".join(str(key) for key in self.default)
            return "{...}" if not keys else "{" + keys + "}"
        return str(self.default)

    def iter_rows(self, *, prefix: str = "") -> tuple[tuple[str, str, str, str, str], ...]:
        path = f"{prefix}{self.key}"
        rows = [
            (
                path,
                self.kind,
                "yes" if self.required else "no",
                self.render_default(),
                self.summary,
            )
        ]
        child_prefix = f"{path}."
        for child in self.children:
            rows.extend(child.iter_rows(prefix=child_prefix))
        return tuple(rows)

    def validate(self, value: Any, *, path: str) -> None:
        if value is None:
            if self.allow_none:
                return
            raise ConfigError(f"{path} must not be null")

        if self.kind == "mapping":
            if not isinstance(value, dict):
                raise ConfigError(f"{path} must be a mapping")
            allowed = {child.key: child for child in self.children}
            unknown = sorted(key for key in value if key not in allowed)
            if unknown and not self.allow_unknown:
                options = ", ".join(sorted(allowed)) or "—"
                raise ConfigError(f"{path} has unknown keys {unknown}. Allowed keys: {options}")
            for child in self.children:
                child_path = f"{path}.{child.key}"
                if child.key not in value:
                    if child.required and not child.has_default:
                        raise ConfigError(f"{child_path} is required")
                    continue
                child.validate(value[child.key], path=child_path)
            return

        if self.kind == "string":
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{path} must be a non-empty string")
            if self.choices and value not in self.choices:
                options = ", ".join(self.choices)
                raise ConfigError(f"{path} must be one of: {options}")
            return

        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ConfigError(f"{path} must be true or false")
            return

        if self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{path} must be a number")
            return

        if self.kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{path} must be an integer")
            return

        if self.kind == "string_list":
            if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
                raise ConfigError(f"{path} must be a list of non-empty strings")
            if self.choices:
                invalid = sorted(item for item in value if item not in self.choices)
                if invalid:
                    options = ", ".join(self.choices)
                    raise ConfigError(f"{path} contains unsupported values {invalid}. Allowed values: {options}")
            return

        if self.kind == "number_list":
            if not isinstance(value, list) or any(
                isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
            ):
                raise ConfigError(f"{path} must be a list of numbers")
            return

        if self.kind == "scalar_list":
            if not isinstance(value, list) or any(isinstance(item, (dict, list)) for item in value):
                raise ConfigError(f"{path} must be a flat list of scalar values")
            return

        if self.kind == "mapping_list":
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise ConfigError(f"{path} must be a list of mappings")
            return

        if self.kind == "scalar":
            if isinstance(value, (dict, list)):
                raise ConfigError(f"{path} must be a scalar value")
            return

        if self.kind == "any":
            return

        raise ConfigError(f"{path} uses unsupported field kind {self.kind!r}")


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
class ProtocolAnalysisChoiceRef:
    key: str
    cases: dict[str, Any]
    default: Any = _UNSET

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("ProtocolAnalysisChoiceRef.key must be a non-empty string.")
        if not isinstance(self.cases, dict) or not self.cases:
            raise ValueError("ProtocolAnalysisChoiceRef.cases must be a non-empty mapping.")
        object.__setattr__(self, "key", self.key.strip())
        object.__setattr__(self, "cases", {str(case_key): deepcopy(value) for case_key, value in self.cases.items()})

    @property
    def has_default(self) -> bool:
        return self.default is not _UNSET


def analysis_choice(key: str, cases: dict[str, Any], default: Any = _UNSET) -> ProtocolAnalysisChoiceRef:
    return ProtocolAnalysisChoiceRef(key=key, cases=cases, default=default)


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
    profiles: tuple[str, ...] = ()
    profile_overrides: dict[str, ProtocolSemanticProfileOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolControlRule.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolControlRule.summary must be a non-empty string.")
        object.__setattr__(self, "match_on", tuple(str(value) for value in self.match_on))
        if self.control_selector is not None and not str(self.control_selector).strip():
            raise ValueError("ProtocolControlRule.control_selector must be a non-empty string when provided.")
        object.__setattr__(self, "profiles", tuple(str(value).strip() for value in self.profiles if str(value).strip()))
        object.__setattr__(self, "profile_overrides", dict(self.profile_overrides or {}))


@dataclass(frozen=True)
class ProtocolWindowSpec:
    id: str
    summary: str
    anchor: str
    selector: str
    params: dict[str, Any] = field(default_factory=dict)
    profiles: tuple[str, ...] = ()
    profile_overrides: dict[str, ProtocolSemanticProfileOverride] = field(default_factory=dict)

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
        object.__setattr__(self, "profiles", tuple(str(value).strip() for value in self.profiles if str(value).strip()))
        object.__setattr__(self, "profile_overrides", dict(self.profile_overrides or {}))


@dataclass(frozen=True)
class ProtocolMetricSpec:
    id: str
    stage: MetricStage
    summary: str
    formula: str
    depends_on: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    profile_overrides: dict[str, ProtocolSemanticProfileOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolMetricSpec.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolMetricSpec.summary must be a non-empty string.")
        if not str(self.formula).strip():
            raise ValueError("ProtocolMetricSpec.formula must be a non-empty string.")
        object.__setattr__(self, "depends_on", tuple(str(value) for value in self.depends_on))
        object.__setattr__(self, "notes", tuple(str(value) for value in self.notes))
        object.__setattr__(self, "profiles", tuple(str(value).strip() for value in self.profiles if str(value).strip()))
        object.__setattr__(self, "profile_overrides", dict(self.profile_overrides or {}))


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
class ProtocolPlotProfileSpec:
    id: str
    summary: str
    figures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolPlotProfileSpec.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolPlotProfileSpec.summary must be a non-empty string.")
        object.__setattr__(self, "figures", tuple(str(value) for value in self.figures))


@dataclass(frozen=True)
class ProtocolArtifactSpec:
    id: str
    summary: str
    default: bool = False

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolArtifactSpec.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolArtifactSpec.summary must be a non-empty string.")


@dataclass(frozen=True)
class ProtocolRankingSpec:
    primary_metric: str
    direction: RankingDirection
    penalties: tuple[str, ...] = ()
    supporting_metrics: tuple[str, ...] = ()
    summary: str = ""
    profiles: tuple[str, ...] = ()
    profile_overrides: dict[str, ProtocolSemanticProfileOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.primary_metric).strip():
            raise ValueError("ProtocolRankingSpec.primary_metric must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolRankingSpec.summary must be a non-empty string.")
        object.__setattr__(self, "penalties", tuple(str(value) for value in self.penalties))
        object.__setattr__(self, "supporting_metrics", tuple(str(value) for value in self.supporting_metrics))
        object.__setattr__(self, "profiles", tuple(str(value).strip() for value in self.profiles if str(value).strip()))
        object.__setattr__(self, "profile_overrides", dict(self.profile_overrides or {}))


@dataclass(frozen=True)
class ProtocolSemanticExecution:
    status: SemanticExecutionStatus = "descriptive_only"
    step_ids: tuple[str, ...] = ()
    plugin_ids: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    config_paths: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_ids", tuple(str(value) for value in self.step_ids if str(value).strip()))
        object.__setattr__(self, "plugin_ids", tuple(str(value) for value in self.plugin_ids if str(value).strip()))
        object.__setattr__(self, "record_ids", tuple(str(value) for value in self.record_ids if str(value).strip()))
        object.__setattr__(
            self,
            "config_paths",
            tuple(str(value) for value in self.config_paths if str(value).strip()),
        )
        object.__setattr__(self, "note", str(self.note).strip())


@dataclass(frozen=True)
class ProtocolSemanticNode:
    id: str
    kind: SemanticNodeKind
    summary: str
    profiles: tuple[str, ...] = ()
    stage: MetricStage | None = None
    formula: str | None = None
    depends_on: tuple[str, ...] = ()
    anchor: str | None = None
    selector: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    match_on: tuple[str, ...] = ()
    control_selector: str | None = None
    primary_metric: str | None = None
    direction: RankingDirection | None = None
    penalties: tuple[str, ...] = ()
    supporting_metrics: tuple[str, ...] = ()
    execution: ProtocolSemanticExecution = field(default_factory=ProtocolSemanticExecution)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("ProtocolSemanticNode.id must be a non-empty string.")
        if not str(self.summary).strip():
            raise ValueError("ProtocolSemanticNode.summary must be a non-empty string.")
        object.__setattr__(self, "profiles", tuple(str(value).strip() for value in self.profiles if str(value).strip()))
        object.__setattr__(self, "depends_on", tuple(str(value) for value in self.depends_on if str(value).strip()))
        object.__setattr__(self, "params", dict(self.params or {}))
        object.__setattr__(self, "match_on", tuple(str(value) for value in self.match_on if str(value).strip()))
        object.__setattr__(self, "penalties", tuple(str(value) for value in self.penalties if str(value).strip()))
        object.__setattr__(
            self,
            "supporting_metrics",
            tuple(str(value) for value in self.supporting_metrics if str(value).strip()),
        )


@dataclass(frozen=True)
class ProtocolSemanticProgram:
    protocol: str
    profiles: tuple[ProtocolSemanticProfileSpec, ...] = ()
    active_profile: str | None = None
    controls: tuple[ProtocolSemanticNode, ...] = ()
    windows: tuple[ProtocolSemanticNode, ...] = ()
    metrics: tuple[ProtocolSemanticNode, ...] = ()
    ranking: ProtocolSemanticNode | None = None

    def __post_init__(self) -> None:
        if not str(self.protocol).strip():
            raise ValueError("ProtocolSemanticProgram.protocol must be a non-empty string.")
        profile_ids: set[str] = set()
        for profile in self.profiles:
            if profile.id in profile_ids:
                raise ValueError(f"Duplicate semantic profile {profile.id!r}.")
            profile_ids.add(profile.id)
        if self.active_profile is not None and self.active_profile not in profile_ids:
            raise ValueError(f"Unknown active semantic profile {self.active_profile!r}.")
        for group_name, nodes in (
            ("controls", self.controls),
            ("windows", self.windows),
            ("metrics", self.metrics),
        ):
            seen: set[str] = set()
            for node in nodes:
                if node.id in seen:
                    raise ValueError(f"Duplicate semantic node {node.id!r} in {group_name}.")
                seen.add(node.id)


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
    semantic_program: ProtocolSemanticProgram | None = None

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
    input_fields: tuple[ProtocolConfigFieldSpec, ...] = ()
    analysis_fields: tuple[ProtocolConfigFieldSpec, ...] = ()
    factors: tuple[ProtocolFactorSpec, ...] = ()
    control_rules: tuple[ProtocolControlRule, ...] = ()
    windows: tuple[ProtocolWindowSpec, ...] = ()
    metrics: tuple[ProtocolMetricSpec, ...] = ()
    effect_signs: tuple[ProtocolEffectSignSpec, ...] = ()
    semantic_profiles: tuple[ProtocolSemanticProfileSpec, ...] = ()
    figures: tuple[ProtocolFigureSpec, ...] = ()
    plot_profiles: tuple[ProtocolPlotProfileSpec, ...] = ()
    default_plot_profile: str | None = None
    artifacts: tuple[ProtocolArtifactSpec, ...] = ()
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
        semantic_profile_ids: set[str] = set()
        for item in self.semantic_profiles:
            if item.id in semantic_profile_ids:
                raise ValueError(f"Duplicate protocol semantic profile {item.id!r}.")
            semantic_profile_ids.add(item.id)
        figure_ids: set[str] = set()
        for item in self.figures:
            if item.id in figure_ids:
                raise ValueError(f"Duplicate protocol figure {item.id!r}.")
            figure_ids.add(item.id)
        plot_profile_ids: set[str] = set()
        for item in self.plot_profiles:
            if item.id in plot_profile_ids:
                raise ValueError(f"Duplicate protocol plot profile {item.id!r}.")
            plot_profile_ids.add(item.id)
            unknown = sorted(set(item.figures) - figure_ids)
            if unknown:
                raise ValueError(f"Protocol plot profile {item.id!r} references unknown figures: {', '.join(unknown)}.")
        if self.default_plot_profile is not None:
            default_plot_profile = str(self.default_plot_profile).strip()
            if not default_plot_profile:
                raise ValueError("ProtocolDescriptor.default_plot_profile must be a non-empty string when provided.")
            if default_plot_profile not in plot_profile_ids:
                raise ValueError(
                    f"ProtocolDescriptor.default_plot_profile {default_plot_profile!r} is not defined in plot_profiles."
                )
            object.__setattr__(self, "default_plot_profile", default_plot_profile)
        artifact_ids: set[str] = set()
        for item in self.artifacts:
            if item.id in artifact_ids:
                raise ValueError(f"Duplicate protocol artifact {item.id!r}.")
            artifact_ids.add(item.id)
        self._validate_semantic_profile_references(semantic_profile_ids)
        self._validate_ranking_metric_references()

    def _validate_semantic_profile_references(self, profile_ids: set[str]) -> None:
        if not profile_ids:
            return
        groups = (
            ("control_rules", self.control_rules),
            ("windows", self.windows),
            ("metrics", self.metrics),
        )
        for group_name, items in groups:
            for item in items:
                unknown = sorted(set(item.profiles) - profile_ids)
                if unknown:
                    raise ValueError(
                        f"ProtocolDescriptor.{group_name} item {item.id!r} references unknown semantic profiles: "
                        f"{', '.join(unknown)}."
                    )
                unknown_overrides = sorted(set(item.profile_overrides) - profile_ids)
                if unknown_overrides:
                    raise ValueError(
                        f"ProtocolDescriptor.{group_name} item {item.id!r} overrides unknown semantic profiles: "
                        f"{', '.join(unknown_overrides)}."
                    )
        if self.ranking is not None:
            unknown = sorted(set(self.ranking.profiles) - profile_ids)
            if unknown:
                raise ValueError(
                    "ProtocolDescriptor.ranking references unknown semantic profiles: " + ", ".join(unknown) + "."
                )
            unknown_overrides = sorted(set(self.ranking.profile_overrides) - profile_ids)
            if unknown_overrides:
                raise ValueError(
                    "ProtocolDescriptor.ranking overrides unknown semantic profiles: "
                    + ", ".join(unknown_overrides)
                    + "."
                )

    def _validate_ranking_metric_references(self) -> None:
        if self.ranking is None:
            return
        metric_ids = {item.id for item in self.metrics}

        def _assert_known(metric_id: str, *, where: str) -> None:
            if metric_id == "domain_defined":
                return
            if metric_id not in metric_ids:
                options = ", ".join(sorted(metric_ids)) or "—"
                raise ValueError(f"{where} references unknown metric {metric_id!r}. Known metrics: {options}.")

        _assert_known(self.ranking.primary_metric, where="ProtocolDescriptor.ranking.primary_metric")
        for metric_id in self.ranking.penalties:
            _assert_known(metric_id, where="ProtocolDescriptor.ranking.penalties")
        for metric_id in self.ranking.supporting_metrics:
            _assert_known(metric_id, where="ProtocolDescriptor.ranking.supporting_metrics")
        for profile_id, override in self.ranking.profile_overrides.items():
            if override.primary_metric is not None:
                _assert_known(
                    override.primary_metric,
                    where=f"ProtocolDescriptor.ranking.profile_overrides[{profile_id!r}].primary_metric",
                )
            if override.penalties is not None:
                for metric_id in override.penalties:
                    _assert_known(
                        metric_id,
                        where=f"ProtocolDescriptor.ranking.profile_overrides[{profile_id!r}].penalties",
                    )
            if override.supporting_metrics is not None:
                for metric_id in override.supporting_metrics:
                    _assert_known(
                        metric_id,
                        where=f"ProtocolDescriptor.ranking.profile_overrides[{profile_id!r}].supporting_metrics",
                    )

    def validate_authoring(self, *, inputs: dict[str, Any], analysis: dict[str, Any]) -> None:
        _validate_protocol_surface(inputs, fields=self.input_fields, path="protocol.inputs", protocol_id=self.protocol)
        _validate_protocol_surface(
            analysis,
            fields=self.analysis_fields,
            path="protocol.analysis",
            protocol_id=self.protocol,
        )

    def semantic_program(self, *, active_profile: str | None = None) -> ProtocolSemanticProgram:
        profile_ids = tuple(item.id for item in self.semantic_profiles)
        if active_profile is not None and active_profile not in profile_ids:
            raise ValueError(f"Unknown semantic profile {active_profile!r} for protocol {self.protocol!r}.")

        def _profile_enabled(
            *,
            profiles: tuple[str, ...],
            profile_overrides: dict[str, ProtocolSemanticProfileOverride],
        ) -> bool:
            if active_profile is None:
                return True
            if profiles and active_profile not in profiles:
                return False
            override = profile_overrides.get(active_profile)
            return not (override is not None and not override.enabled)

        def _override(
            profile_overrides: dict[str, ProtocolSemanticProfileOverride],
        ) -> ProtocolSemanticProfileOverride | None:
            if active_profile is None:
                return None
            return profile_overrides.get(active_profile)

        def _node_profiles(
            profiles: tuple[str, ...],
            profile_overrides: dict[str, ProtocolSemanticProfileOverride],
        ) -> tuple[str, ...]:
            if profiles:
                return profiles
            if profile_ids:
                return profile_ids
            if profile_overrides:
                return tuple(sorted(profile_overrides))
            return ()

        return ProtocolSemanticProgram(
            protocol=self.protocol,
            profiles=self.semantic_profiles,
            active_profile=active_profile,
            controls=tuple(
                ProtocolSemanticNode(
                    id=item.id,
                    kind="control_rule",
                    summary=(
                        _override(item.profile_overrides).summary
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).summary is not None
                        else item.summary
                    ),
                    profiles=_node_profiles(item.profiles, item.profile_overrides),
                    match_on=(
                        _override(item.profile_overrides).match_on
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).match_on is not None
                        else item.match_on
                    ),
                    control_selector=(
                        _override(item.profile_overrides).control_selector
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).control_selector is not None
                        else item.control_selector
                    ),
                )
                for item in self.control_rules
                if _profile_enabled(profiles=item.profiles, profile_overrides=item.profile_overrides)
            ),
            windows=tuple(
                ProtocolSemanticNode(
                    id=item.id,
                    kind="window",
                    summary=(
                        _override(item.profile_overrides).summary
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).summary is not None
                        else item.summary
                    ),
                    profiles=_node_profiles(item.profiles, item.profile_overrides),
                    anchor=(
                        _override(item.profile_overrides).anchor
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).anchor is not None
                        else item.anchor
                    ),
                    selector=(
                        _override(item.profile_overrides).selector
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).selector is not None
                        else item.selector
                    ),
                    params=(
                        _override(item.profile_overrides).params
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).params is not None
                        else item.params
                    ),
                )
                for item in self.windows
                if _profile_enabled(profiles=item.profiles, profile_overrides=item.profile_overrides)
            ),
            metrics=tuple(
                ProtocolSemanticNode(
                    id=item.id,
                    kind="metric",
                    summary=(
                        _override(item.profile_overrides).summary
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).summary is not None
                        else item.summary
                    ),
                    profiles=_node_profiles(item.profiles, item.profile_overrides),
                    stage=item.stage,
                    formula=(
                        _override(item.profile_overrides).formula
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).formula is not None
                        else item.formula
                    ),
                    depends_on=(
                        _override(item.profile_overrides).depends_on
                        if _override(item.profile_overrides) is not None
                        and _override(item.profile_overrides).depends_on is not None
                        else item.depends_on
                    ),
                )
                for item in self.metrics
                if _profile_enabled(profiles=item.profiles, profile_overrides=item.profile_overrides)
            ),
            ranking=(
                ProtocolSemanticNode(
                    id="ranking",
                    kind="ranking",
                    summary=(
                        _override(self.ranking.profile_overrides).summary
                        if _override(self.ranking.profile_overrides) is not None
                        and _override(self.ranking.profile_overrides).summary is not None
                        else self.ranking.summary
                    ),
                    profiles=_node_profiles(self.ranking.profiles, self.ranking.profile_overrides),
                    primary_metric=(
                        _override(self.ranking.profile_overrides).primary_metric
                        if _override(self.ranking.profile_overrides) is not None
                        and _override(self.ranking.profile_overrides).primary_metric is not None
                        else self.ranking.primary_metric
                    ),
                    direction=(
                        _override(self.ranking.profile_overrides).direction
                        if _override(self.ranking.profile_overrides) is not None
                        and _override(self.ranking.profile_overrides).direction is not None
                        else self.ranking.direction
                    ),
                    penalties=(
                        _override(self.ranking.profile_overrides).penalties
                        if _override(self.ranking.profile_overrides) is not None
                        and _override(self.ranking.profile_overrides).penalties is not None
                        else self.ranking.penalties
                    ),
                    supporting_metrics=(
                        _override(self.ranking.profile_overrides).supporting_metrics
                        if _override(self.ranking.profile_overrides) is not None
                        and _override(self.ranking.profile_overrides).supporting_metrics is not None
                        else self.ranking.supporting_metrics
                    ),
                )
                if self.ranking is not None
                and _profile_enabled(
                    profiles=self.ranking.profiles,
                    profile_overrides=self.ranking.profile_overrides,
                )
                else None
            ),
        )


@dataclass(frozen=True)
class BoundProtocol:
    descriptor: ProtocolDescriptor
    inputs: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", dict(self.inputs or {}))
        object.__setattr__(self, "analysis", dict(self.analysis or {}))
        object.__setattr__(self, "outputs", dict(self.outputs or {}))
        self.descriptor.validate_authoring(inputs=self.inputs, analysis=self.analysis)

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
    def default_plot_profile(self) -> str | None:
        return self.descriptor.default_plot_profile

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
            semantic_program=plan.semantic_program or self.descriptor.semantic_program(),
        )

    def effective_plugin_config(self, *, plugin_id: str, step_with: dict[str, Any] | None = None) -> dict[str, Any]:
        defaults = self._protocol_plugin_defaults(plugin_id)
        return _deep_merge(defaults, dict(step_with or {}))

    def configured_notebook_template(self) -> str | None:
        block = self._output_block("notebook")
        template = block.get("template")
        if template is None:
            return None
        if not isinstance(template, str) or not template.strip():
            raise ConfigError(f"protocol.outputs.notebook.template for {self.id!r} must be a non-empty string")
        return template.strip()

    def select_plot_outputs(
        self,
        *,
        default_profile: str | None = None,
        allowed: set[str],
    ) -> tuple[str, ...]:
        block = self._output_block("plots")
        profile = block.get("profile", default_profile or self.descriptor.default_plot_profile or "none")
        profile_ids = self._plot_profile_members(profile=profile, allowed=allowed)
        include = self._validate_deliverable_ids(
            block.get("include", ()),
            where="protocol.outputs.plots.include",
            allowed=allowed,
        )
        exclude = set(
            self._validate_deliverable_ids(
                block.get("exclude", ()),
                where="protocol.outputs.plots.exclude",
                allowed=allowed,
            )
        )
        for figure_id in self._validate_named_output_configs(section="plots", key="views", allowed=allowed):
            if figure_id not in allowed:
                options = ", ".join(sorted(allowed)) or "—"
                raise ConfigError(
                    f"protocol.outputs.plots.views.{figure_id!r} is unknown for {self.id!r}. Available ids: {options}"
                )
        selected: list[str] = []
        for figure_id in (*profile_ids, *include):
            if figure_id in exclude:
                continue
            if figure_id not in selected:
                selected.append(figure_id)
        return tuple(selected)

    def plot_view_config(self, *, figure_id: str) -> dict[str, Any]:
        block = self._output_block("plots")
        views = block.get("views", {})
        if not isinstance(views, dict):
            raise ConfigError(f"protocol.outputs.plots.views for {self.id!r} must be a mapping")
        configured = views.get(figure_id, {})
        if configured is None:
            return {}
        if not isinstance(configured, dict):
            raise ConfigError(f"protocol.outputs.plots.views.{figure_id!r} for {self.id!r} must be a mapping")
        return deepcopy(configured)

    def select_export_outputs(
        self,
        *,
        defaults: tuple[str, ...],
        allowed: set[str],
    ) -> tuple[str, ...]:
        block = self._output_block("exports")
        include = self._validate_deliverable_ids(
            block.get("include", ()),
            where="protocol.outputs.exports.include",
            allowed=allowed,
        )
        exclude = set(
            self._validate_deliverable_ids(
                block.get("exclude", ()),
                where="protocol.outputs.exports.exclude",
                allowed=allowed,
            )
        )
        for artifact_id in self._validate_named_output_configs(section="exports", key="artifacts", allowed=allowed):
            if artifact_id not in allowed:
                options = ", ".join(sorted(allowed)) or "—"
                raise ConfigError(
                    f"protocol.outputs.exports.artifacts.{artifact_id!r} is unknown for {self.id!r}. "
                    f"Available ids: {options}"
                )
        selected: list[str] = []
        for artifact_id in (*defaults, *include):
            if artifact_id in exclude:
                continue
            if artifact_id not in selected:
                selected.append(artifact_id)
        return tuple(selected)

    def export_artifact_config(self, *, artifact_id: str) -> dict[str, Any]:
        block = self._output_block("exports")
        artifacts = block.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ConfigError(f"protocol.outputs.exports.artifacts for {self.id!r} must be a mapping")
        configured = artifacts.get(artifact_id, {})
        if configured is None:
            return {}
        if not isinstance(configured, dict):
            raise ConfigError(f"protocol.outputs.exports.artifacts.{artifact_id!r} for {self.id!r} must be a mapping")
        return deepcopy(configured)

    def _protocol_plugin_defaults(self, plugin_id: str) -> dict[str, Any]:
        for item in self.execution.plugin_defaults:
            if item.plugin == plugin_id:
                return self._resolve_binding_refs(item.with_, where=f"protocol {self.id} plugin {plugin_id}")
        return {}

    def _output_block(self, section: str) -> dict[str, Any]:
        raw = self.outputs.get(section, {})
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ConfigError(f"protocol.outputs.{section} for {self.id!r} must be a mapping")
        return dict(raw)

    def _plot_profile_members(self, *, profile: Any, allowed: set[str]) -> tuple[str, ...]:
        if not isinstance(profile, str) or not profile.strip():
            raise ConfigError(f"protocol.outputs.plots.profile for {self.id!r} must be a non-empty string")
        profile_id = profile.strip()
        if profile_id == "none":
            return ()
        for item in self.descriptor.plot_profiles:
            if item.id == profile_id:
                members = tuple(figure_id for figure_id in item.figures if figure_id in allowed)
                if len(members) != len(item.figures):
                    unknown = sorted(set(item.figures) - allowed)
                    raise ConfigError(
                        f"protocol.outputs.plots.profile {profile_id!r} includes unsupported figures "
                        f"for {self.id!r}: {', '.join(unknown)}"
                    )
                return members
        options = ", ".join(["none", *[item.id for item in self.descriptor.plot_profiles]]) or "—"
        raise ConfigError(f"protocol.outputs.plots.profile must be one of: {options}")

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

    def _validate_named_output_configs(self, *, section: str, key: str, allowed: set[str]) -> tuple[str, ...]:
        block = self._output_block(section)
        settings = block.get(key, {})
        if settings in (None, {}):
            return ()
        if not isinstance(settings, dict):
            raise ConfigError(f"protocol.outputs.{section}.{key} for {self.id!r} must be a mapping")
        ids: list[str] = []
        for deliverable_id, config in settings.items():
            if not isinstance(deliverable_id, str) or not deliverable_id.strip():
                raise ConfigError(f"protocol.outputs.{section}.{key} for {self.id!r} must use non-empty ids")
            if not isinstance(config, dict):
                raise ConfigError(
                    f"protocol.outputs.{section}.{key}.{deliverable_id!r} for {self.id!r} must be a mapping"
                )
            if deliverable_id.strip() not in allowed:
                options = ", ".join(sorted(allowed)) or "—"
                raise ConfigError(
                    f"protocol.outputs.{section}.{key}.{deliverable_id!r} is unknown for {self.id!r}. "
                    f"Available ids: {options}"
                )
            ids.append(deliverable_id.strip())
        return tuple(ids)

    def _resolve_binding_refs(self, value: Any, *, where: str) -> Any:
        if isinstance(value, ProtocolBindingValueRef):
            found, resolved = self._lookup_parameter_value(value.key)
            if found:
                return deepcopy(resolved)
            if value.has_default:
                return self._resolve_binding_refs(value.default, where=where)
            raise ConfigError(f"{where} requires protocol.inputs.{value.key}")
        if isinstance(value, ProtocolAnalysisChoiceRef):
            found, resolved = self._lookup_analysis_value(value.key)
            if found:
                selected = value.cases.get(str(resolved))
                if selected is not None:
                    return deepcopy(selected)
            if value.has_default:
                return deepcopy(value.default)
            raise ConfigError(f"{where} requires protocol.analysis.{value.key}")
        if isinstance(value, dict):
            return {key: self._resolve_binding_refs(item, where=where) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_binding_refs(item, where=where) for item in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_binding_refs(item, where=where) for item in value)
        return deepcopy(value)

    def _lookup_parameter_value(self, key: str) -> tuple[bool, Any]:
        current: Any = self.inputs
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current

    def _lookup_analysis_value(self, key: str) -> tuple[bool, Any]:
        current: Any = self.analysis
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current

    def authoring_rows(self, *, section: Literal["inputs", "analysis"]) -> tuple[tuple[str, str, str, str, str], ...]:
        fields = self.descriptor.input_fields if section == "inputs" else self.descriptor.analysis_fields
        rows: list[tuple[str, str, str, str, str]] = []
        for field_spec in fields:
            rows.extend(field_spec.iter_rows())
        return tuple(rows)


def _deep_merge(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = _deep_merge(merged[key], value)
                continue
            merged[key] = deepcopy(value)
    return merged


def _validate_protocol_surface(
    raw: dict[str, Any],
    *,
    fields: tuple[ProtocolConfigFieldSpec, ...],
    path: str,
    protocol_id: str,
) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} for {protocol_id!r} must be a mapping")
    allowed = {field.key: field for field in fields}
    unknown = sorted(key for key in raw if key not in allowed)
    if unknown:
        options = ", ".join(sorted(allowed)) or "—"
        raise ConfigError(f"{path} for {protocol_id!r} has unknown keys {unknown}. Allowed keys: {options}")
    for field_spec in fields:
        field_path = f"{path}.{field_spec.key}"
        if field_spec.key not in raw:
            if field_spec.required and not field_spec.has_default:
                raise ConfigError(f"{field_path} for {protocol_id!r} is required")
            continue
        field_spec.validate(raw[field_spec.key], path=field_path)


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
            inputs=binding.inputs,
            analysis=binding.analysis,
            outputs=binding.outputs,
        )

    def list(self, *, domain: str | None = None, family: str | None = None) -> list[tuple[str, str]]:
        return [
            (item.protocol, item.summary)
            for item in self._descriptors
            if (domain is None or item.domain == domain) and (family is None or item.family == family)
        ]
