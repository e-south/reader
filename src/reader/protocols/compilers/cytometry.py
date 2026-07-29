from __future__ import annotations

from typing import Any

from reader.errors import ConfigError
from reader.protocols.model import CompiledProtocolPlan
from reader.protocols.semantic_coverage import _cytometry_semantic_program
from reader.workbench.decl.model import RecordInputDecl, RecordOutputDecl, ResourceInputDecl

from .common import _step, default_notebook_call


def compile_cytometry_flow_panel(protocol: Any):
    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    try:
        selected_plots = protocol.select_plot_outputs(allowed=set())
    except ConfigError as exc:
        raise ConfigError(f"cytometry/flow_panel does not currently compile plot outputs. {exc}") from exc
    try:
        selected_exports = protocol.select_export_outputs(defaults=(), allowed=set())
    except ConfigError as exc:
        raise ConfigError(f"cytometry/flow_panel does not currently compile export artifacts. {exc}") from exc
    if selected_plots:
        raise ConfigError("cytometry/flow_panel does not currently compile plot outputs.")
    if selected_exports:
        raise ConfigError("cytometry/flow_panel does not currently compile export artifacts.")
    return CompiledProtocolPlan(
        pipeline=(
            _step(
                id="ingest_cytometer",
                plugin="ingest/flow_cytometer",
                writes={"df": RecordOutputDecl(record_id="ingest/df")},
            ),
            _step(
                id="merge_metadata",
                plugin="transform/sample_metadata",
                reads={
                    "df": RecordInputDecl(record_id="ingest/df"),
                    "metadata": ResourceInputDecl(resource_id="metadata"),
                },
                writes={"df": RecordOutputDecl(record_id="merged/df")},
            ),
        ),
        plots=(),
        exports=(),
        notebooks=(default_notebook_call(template),),
        semantic_program=_cytometry_semantic_program(protocol),
    )
