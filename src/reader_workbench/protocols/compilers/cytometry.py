from __future__ import annotations

from typing import Any

from reader_workbench.protocols.model import CompiledProtocolPlan
from reader_workbench.protocols.semantic_coverage import _cytometry_semantic_program
from reader_workbench.workbench.decl.model import RecordInputDecl, RecordOutputDecl, ResourceInputDecl

from .common import _deep_merge, _step

CYTOMETRY_PLOT_OUTPUTS = {"gating_diagnostic"}
CYTOMETRY_EXPORT_OUTPUTS = {
    "gate_definition_table": ("cytometry_gating/gate_definition", "cytometry_gate_definition.csv"),
    "sample_stats_table": ("cytometry_gating/sample_stats", "cytometry_sample_stats.csv"),
    "group_stats_table": ("cytometry_gating/group_stats", "cytometry_group_stats.csv"),
    "qc_table": ("cytometry_gating/qc", "cytometry_qc.csv"),
    "gated_events_table": ("cytometry_gating/gated_events", "cytometry_gated_events.csv"),
}


def compile_cytometry_flow_panel(protocol: Any):
    selected_plots = protocol.select_plot_outputs(allowed=CYTOMETRY_PLOT_OUTPUTS)
    selected_exports = protocol.select_export_outputs(
        defaults=("gate_definition_table", "sample_stats_table", "group_stats_table", "qc_table"),
        allowed=set(CYTOMETRY_EXPORT_OUTPUTS),
    )
    pipeline = (
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
        _step(
            id="cytometry_gating",
            plugin="transform/cytometry_gating",
            reads={"events": RecordInputDecl(record_id="merged/df")},
            writes={
                "gate_definition": RecordOutputDecl(record_id="cytometry_gating/gate_definition"),
                "gated_events": RecordOutputDecl(record_id="cytometry_gating/gated_events"),
                "sample_stats": RecordOutputDecl(record_id="cytometry_gating/sample_stats"),
                "group_stats": RecordOutputDecl(record_id="cytometry_gating/group_stats"),
                "qc": RecordOutputDecl(record_id="cytometry_gating/qc"),
            },
        ),
    )
    plots = tuple(
        _step(
            id="gating_diagnostic",
            plugin="plot/cytometry_diagnostic",
            reads={
                "original_events": RecordInputDecl(record_id="merged/df"),
                "gate_definition": RecordInputDecl(record_id="cytometry_gating/gate_definition"),
                "gated_events": RecordInputDecl(record_id="cytometry_gating/gated_events"),
            },
            with_=protocol.plot_view_config(figure_id="gating_diagnostic"),
        )
        for _ in selected_plots
    )
    exports = tuple(_cytometry_export_output(protocol, output_id=output_id) for output_id in selected_exports)
    return CompiledProtocolPlan(
        pipeline=pipeline,
        plots=plots,
        exports=exports,
        semantic_program=_cytometry_semantic_program(protocol),
    )


def _cytometry_export_output(protocol: Any, *, output_id: str):
    record_id, default_path = CYTOMETRY_EXPORT_OUTPUTS[output_id]
    return _step(
        id=output_id,
        plugin="export/csv",
        reads={"df": RecordInputDecl(record_id=record_id)},
        with_=_deep_merge(
            {"path": default_path},
            protocol.export_artifact_config(artifact_id=output_id),
        ),
    )
