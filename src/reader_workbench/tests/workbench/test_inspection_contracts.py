from __future__ import annotations

from typing import cast

import pytest
from rich.console import Console

import reader_workbench.workbench.decl as decl_module
from reader_workbench.contracts import builtin_contract_catalog
from reader_workbench.protocols import (
    CompiledProtocolPlan,
    ProtocolCatalog,
    ProtocolConfigFieldSpec,
    ProtocolDescriptor,
    ProtocolExecutionPlan,
)
from reader_workbench.runtime import ReaderRuntime, builtin_runtime
from reader_workbench.tests.support import base_reader_config, build_decl, write_config
from reader_workbench.workbench.cli import THEME
from reader_workbench.workbench.config import ReaderSpec
from reader_workbench.workbench.experiment import ResourceEntry
from reader_workbench.workbench.inspection.experiments import (
    _resource_entry_payload,
    experiment_config_json_payload,
    experiment_explain_payload,
    experiment_inspect_payload,
    experiment_steps_payload,
)
from reader_workbench.workbench.inspection.protocols import protocol_descriptor_payload
from reader_workbench.workbench.inspection.reports import experiment_inspect_renderables
from reader_workbench.workbench.inspection.runtime import serialize_reads
from reader_workbench.workbench.registry import Registry


def test_decl_public_exports_all_resolve() -> None:
    assert "bind_recipe_steps" not in decl_module.__all__
    assert {name: getattr(decl_module, name) for name in decl_module.__all__}


def test_compiled_inspection_serializes_record_resource_collections() -> None:
    reads = {"sources": decl_module.RecordCollectionInputDecl(resource_ids=("source_a", "source_b"))}

    assert serialize_reads(reads) == [
        {
            "label": "sources",
            "display": "record_resources(source_a, source_b)",
            "ref": {"record_resources": ["source_a", "source_b"]},
        }
    ]


def test_experiment_inspection_serializes_record_resources_by_identity(tmp_path) -> None:
    source_root = tmp_path / "experiments" / "2026" / "source-a"
    source_root.mkdir(parents=True)
    write_config(source_root, base_reader_config(experiment_id="source-a"))
    aggregate_root = tmp_path / "experiments" / "aggregates" / "aggregate-a"
    aggregate_root.mkdir(parents=True)
    config_path = write_config(
        aggregate_root,
        base_reader_config(
            experiment_id="aggregate-a",
            protocol_id="logic/four_state_vector_collection",
            protocol_inputs={"record_resources": ["source"]},
            resources={
                "metadata": {"kind": "file", "path": "./inputs/metadata.csv"},
                "source": {
                    "kind": "record",
                    "experiment": "source-a",
                    "record": "vector/df",
                },
            },
        ),
    )
    spec = ReaderSpec.load(config_path)
    declaration = build_decl(spec, source_path=config_path)

    payload = experiment_inspect_payload(
        job_path=config_path,
        spec=spec,
        decl=declaration,
        runtime=builtin_runtime(),
    )

    assert payload["implementation"]["inputs"]["resources"] == [
        {
            "id": "metadata",
            "kind": "file",
            "path": "inputs/metadata.csv",
        },
        {
            "id": "source",
            "kind": "record",
            "experiment": "source-a",
            "record": "vector/df",
        },
    ]
    console = Console(record=True, width=120, theme=THEME)
    for renderable in experiment_inspect_renderables(
        payload=payload,
        semantic_program=declaration.experiment_semantics.protocol_program,
    ):
        console.print(renderable)
    assert "source-a:vector/df" in console.export_text()


def test_experiment_inspection_rejects_unsupported_resource_entries(tmp_path) -> None:
    with pytest.raises(TypeError, match="Unsupported experiment resource entry: object"):
        _resource_entry_payload("unsupported", cast(ResourceEntry, object()), base=tmp_path)


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
