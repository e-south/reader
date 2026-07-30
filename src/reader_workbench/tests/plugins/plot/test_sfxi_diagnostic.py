from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import ValidationError

from reader_workbench.errors import ExecutionError
from reader_workbench.plugins.plot.sfxi_diagnostic import SFXIDiagnosticCfg, SFXIDiagnosticPlot
from reader_workbench.protocols import ProtocolBinding, ProtocolSemanticProgram
from reader_workbench.runtime import builtin_runtime
from reader_workbench.workbench.decl.model import (
    ExperimentDecl,
    PipelineDecl,
    PluginStepDecl,
    RecordInputDecl,
    SurfaceDecl,
    WorkbenchDecl,
)
from reader_workbench.workbench.engine import run_spec
from reader_workbench.workbench.experiment import (
    AnnotationSemantics,
    ExperimentSemantics,
    OrderedStateSpaces,
    OrderedStateSpaceSpec,
    OutputLayout,
    ResourceCatalog,
)
from reader_workbench.workbench.graph import resolve_workbench


def _context():
    return SimpleNamespace(
        experiment=SimpleNamespace(
            annotations=AnnotationSemantics(
                ordered_state_spaces=OrderedStateSpaces(
                    by_id={
                        "states": OrderedStateSpaceSpec(
                            column="condition",
                            state_order=("00", "10", "01", "11"),
                            source_values={"00": "off", "10": "a", "01": "b", "11": "both"},
                        )
                    }
                )
            )
        )
    )


def _inputs() -> dict[str, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for design_id in ("design/a", "design-b"):
        for state_index, condition in enumerate(("off", "a", "b", "both")):
            for time in (0.0, 1.0):
                for channel in ("OD600", "response"):
                    rows.append(
                        {
                            "design_id": design_id,
                            "condition": condition,
                            "time": time,
                            "channel": channel,
                            "value": float(state_index + time + (channel == "response")),
                            "position": "A1",
                        }
                    )
    vec8 = pd.DataFrame.from_records(
        [
            {
                "design_id": design_id,
                "time_selected_h": 1.0,
                "reference_design_id": "reference",
                "intensity_log2_offset_delta": 0.0,
                "r_logic": 4.0,
                "v00": 0.0,
                "v10": 0.25,
                "v01": 0.75,
                "v11": 1.0,
                "y00_star": -1.0,
                "y10_star": -0.5,
                "y01_star": 0.5,
                "y11_star": 1.0,
                "flat_logic": False,
            }
            for design_id in ("design/a", "design-b")
        ]
    )
    return {"df": pd.DataFrame.from_records(rows), "vec8": vec8}


def test_sfxi_diagnostic_declares_both_persisted_inputs() -> None:
    ports = SFXIDiagnosticPlot.input_ports()

    assert ports["df"].contract == "plate_reader.annotated.v1"
    assert ports["vec8"].contract == "sfxi.vec8.v3"


@pytest.mark.parametrize("legacy_key", ["trajectory_ci", "trajectory_bootstraps"])
def test_sfxi_diagnostic_rejects_legacy_interval_keys(legacy_key: str) -> None:
    with pytest.raises(ValidationError, match=legacy_key):
        SFXIDiagnosticCfg(state_map_ref="states", **{legacy_key: 10})


def test_sfxi_diagnostic_defaults_to_one_artifact_per_persisted_design() -> None:
    figures = SFXIDiagnosticPlot().render(
        _context(),
        _inputs(),
        SFXIDiagnosticCfg(
            state_map_ref="states",
            response_channel="response",
            format=["png"],
            dpi=72,
        ),
    )

    assert [item.filename for item in figures] == [
        "sfxi_diagnostic--design_a",
        "sfxi_diagnostic--design-b",
    ]
    assert all(item.ext == "png" for item in figures)
    assert all(item.description and "persisted vec8" in item.description for item in figures)
    for item in figures:
        plt.close(item.fig)


def test_sfxi_diagnostic_rejects_colliding_design_filename_slugs() -> None:
    inputs = _inputs()
    inputs["df"]["design_id"] = inputs["df"]["design_id"].replace({"design-b": "design_a"})
    inputs["vec8"]["design_id"] = inputs["vec8"]["design_id"].replace({"design-b": "design_a"})

    with pytest.raises(ExecutionError, match="duplicate artifact filenames"):
        SFXIDiagnosticPlot().render(
            _context(),
            inputs,
            SFXIDiagnosticCfg(
                state_map_ref="states",
                response_channel="response",
                format=["png"],
                dpi=72,
            ),
        )


def test_sfxi_diagnostic_runtime_persists_one_normal_plot_bundle(tmp_path: Path) -> None:
    runtime = builtin_runtime()
    outputs = tmp_path / "outputs"
    store = runtime.record_store(outputs, plots_subdir="plots", exports_subdir="exports")
    inputs = _inputs()
    annotated = inputs["df"].copy()
    annotated["treatment"] = annotated["condition"]
    store.persist_dataframe(
        producer_id="promote_to_tidy_plus_map",
        producer_plugin="validator/to_tidy_plus_map",
        out_name="df",
        record_id="promote_to_tidy_plus_map/df",
        df=annotated,
        contract_id="plate_reader.annotated.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    store.persist_dataframe(
        producer_id="sfxi_vec8",
        producer_plugin="transform/sfxi",
        out_name="vec8",
        record_id="sfxi_vec8/vec8",
        df=inputs["vec8"],
        contract_id="sfxi.vec8.v3",
        inputs=[],
        config_digest="sha256:test",
    )
    annotations = _context().experiment.annotations
    declaration = WorkbenchDecl(
        experiment=ExperimentDecl(id="sfxi-diagnostic", title="SFXI diagnostic", lifecycle="active", root=tmp_path),
        experiment_semantics=ExperimentSemantics(
            protocol=ProtocolBinding(id="workbench/generic"),
            protocol_program=ProtocolSemanticProgram(protocol="workbench/generic"),
            annotations=annotations,
            resources=ResourceCatalog(),
            layout=OutputLayout(
                outputs_dir=outputs,
                plots_subdir="plots",
                exports_subdir="exports",
                notebooks_subdir="notebooks",
            ),
        ),
        plotting_palette=None,
        pipeline=PipelineDecl(runtime={}, steps=()),
        plots=SurfaceDecl(
            specs=(
                PluginStepDecl(
                    id="sfxi_diagnostic",
                    plugin="plot/sfxi_diagnostic",
                    reads={
                        "df": RecordInputDecl(record_id="promote_to_tidy_plus_map/df"),
                        "vec8": RecordInputDecl(record_id="sfxi_vec8/vec8"),
                    },
                    with_={
                        "state_map_ref": "states",
                        "response_channel": "response",
                        "design_ids": ["design-b"],
                        "trajectory_resamples": 10,
                        "format": ["png"],
                        "dpi": 72,
                    },
                ),
            )
        ),
        exports=SurfaceDecl(specs=()),
    )

    run_spec(
        declaration,
        include_pipeline=False,
        include_plots=True,
        include_exports=False,
        plot_specs=resolve_workbench(declaration).plots,
        log_level="ERROR",
        runtime=runtime,
    )

    record = store.latest_record("plot:sfxi_diagnostic")
    assert record is not None
    assert {item.ref.record_id for item in record.inputs} == {
        "promote_to_tidy_plus_map/df",
        "sfxi_vec8/vec8",
    }
    assert (outputs / "plots" / "sfxi_diagnostic--design-b.png").is_file()
