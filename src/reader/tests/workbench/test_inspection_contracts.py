from __future__ import annotations

from rich.console import Console

import reader.workbench.decl as decl_module
from reader.contracts import builtin_contract_catalog
from reader.protocols import (
    CompiledProtocolPlan,
    ProtocolCatalog,
    ProtocolConfigFieldSpec,
    ProtocolDescriptor,
    ProtocolExecutionPlan,
)
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.tests.support import base_reader_config, build_decl, write_config
from reader.workbench.cli import THEME
from reader.workbench.config import ReaderSpec
from reader.workbench.inspection.experiments import (
    experiment_config_json_payload,
    experiment_explain_payload,
    experiment_inspect_payload,
    experiment_steps_payload,
)
from reader.workbench.inspection.protocols import protocol_descriptor_payload
from reader.workbench.inspection.reports import experiment_inspect_renderables
from reader.workbench.registry import Registry


def test_decl_public_exports_all_resolve() -> None:
    assert "bind_recipe_steps" not in decl_module.__all__
    assert {name: getattr(decl_module, name) for name in decl_module.__all__}


def test_experiment_inspection_payloads_do_not_advertise_notebook_planning(tmp_path) -> None:
    config_path = write_config(tmp_path / "config.yaml", base_reader_config(experiment_id="inspection_contract"))
    spec = ReaderSpec.load(config_path)
    declaration = build_decl(spec, source_path=config_path)
    runtime = builtin_runtime()

    payloads = (
        experiment_explain_payload(job_path=config_path, spec=spec, decl=declaration, runtime=runtime),
        experiment_steps_payload(job_path=config_path, spec=spec, decl=declaration, runtime=runtime),
        experiment_config_json_payload(job_path=config_path, spec=spec, decl=declaration, runtime=runtime),
        experiment_inspect_payload(job_path=config_path, spec=spec, decl=declaration, runtime=runtime),
    )

    for payload in payloads:
        assert "notebook_template" not in payload["experiment"]
        assert "notebooks" not in payload["implementation"]["plan"]
        assert "notebooks" not in payload["implementation"]["compiled"]


def test_protocol_inspection_does_not_advertise_notebook_policy_when_authoring_is_required() -> None:
    descriptor = ProtocolDescriptor(
        protocol="generic/requires_authoring",
        domain="generic",
        family="inspection_contract",
        summary="Synthetic protocol that requires explicit authoring.",
        input_fields=(
            ProtocolConfigFieldSpec(
                key="source",
                summary="Required synthetic input.",
                kind="string",
                required=True,
            ),
        ),
        execution=ProtocolExecutionPlan(
            compiler=lambda protocol: CompiledProtocolPlan(semantic_program=protocol.semantic_program()),
        ),
    )
    contracts = builtin_contract_catalog()
    runtime = ReaderRuntime(
        contracts=contracts,
        protocols=ProtocolCatalog([descriptor]),
        plugins=Registry(contracts=contracts),
    )

    payload = protocol_descriptor_payload(descriptor, runtime=runtime)

    assert "notebook_policy" not in payload["authoring"]["outputs"]
    assert "notebook" not in payload["authoring"]["starter_config"]["protocol"]["outputs"]
    assert payload["implementation"]["compiled"]["status"] == "requires_authoring"
    assert "notebooks" not in payload["implementation"]["compiled"]


def test_experiment_inspection_report_has_no_template_selection_panel(tmp_path) -> None:
    config_path = write_config(tmp_path / "config.yaml", base_reader_config(experiment_id="inspection_report"))
    spec = ReaderSpec.load(config_path)
    declaration = build_decl(spec, source_path=config_path)
    payload = experiment_explain_payload(
        job_path=config_path,
        spec=spec,
        decl=declaration,
        runtime=builtin_runtime(),
    )
    console = Console(record=True, width=120, theme=THEME)

    for renderable in experiment_inspect_renderables(
        payload=payload,
        semantic_program=declaration.experiment_semantics.protocol_program,
    ):
        console.print(renderable)

    rendered = console.export_text()
    assert "No notebook template selected" not in rendered
    assert "Notebooks" not in rendered
