from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reader_workbench.contracts import builtin_contract_catalog
from reader_workbench.domains.logic.four_state_vector.heatmap import normalize_experiment_heatmap_frame
from reader_workbench.errors import FourStateVectorError
from reader_workbench.plugins.plot.four_state_vector_heatmap import (
    FourStateVectorHeatmapCfg,
    FourStateVectorHeatmapPlot,
)
from reader_workbench.protocols import ProtocolBinding, ProtocolSemanticProgram
from reader_workbench.runtime import builtin_runtime
from reader_workbench.workbench import PluginSemantics
from reader_workbench.workbench.assets import build_plugin_asset
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
    OutputLayout,
    ResourceCatalog,
)
from reader_workbench.workbench.graph import resolve_workbench


def _vector_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": ["design-2", "design-10"],
            "reference_design_id": ["reference", "reference"],
            "time_selected_h": [12.0, 12.0],
            "intensity_log2_offset_delta": [0.0, 0.0],
            "r_logic": [8.0, 4.0],
            "v00": [0.0, 0.0],
            "v10": [0.7, 0.1],
            "v01": [0.2, 0.1],
            "v11": [1.0, 1.0],
            "y00_star": [-1.0, -0.5],
            "y10_star": [1.0, 0.0],
            "y01_star": [0.3, 0.2],
            "y11_star": [4.0, 2.0],
            "flat_logic": [False, False],
        }
    )


def test_four_state_vector_heatmap_plot_saves_artifact(tmp_path: Path) -> None:
    ctx = SimpleNamespace(plots_dir=tmp_path, exp_dir=tmp_path / "experiment-2")
    cfg = FourStateVectorHeatmapCfg(experiment_id="experiment-2", format=["png"], title="four-state vector heatmap")
    plugin = FourStateVectorHeatmapPlot()
    plugin.bind_runtime(
        descriptor=build_plugin_asset(
            plugin_id="plot/four_state_vector_heatmap",
            semantics=PluginSemantics(
                domain="logic",
                family="four_state_vector_heatmap",
                summary="four-state vector heatmap plot.",
            ),
            plugin_cls=FourStateVectorHeatmapPlot,
        ),
        contracts=builtin_contract_catalog(),
    )

    output = plugin.run(ctx, {"vector": _vector_df()}, cfg)

    assert output["artifacts"] == [str(tmp_path / "four_state_vector_heatmap.png")]
    assert (tmp_path / "four_state_vector_heatmap.png").exists()


def test_normalize_experiment_heatmap_frame_rejects_missing_intensity_delta() -> None:
    with pytest.raises(FourStateVectorError, match="requires column 'intensity_log2_offset_delta'"):
        normalize_experiment_heatmap_frame(
            _vector_df().drop(columns=["intensity_log2_offset_delta"]),
            experiment_id="experiment-2",
        )


def test_normalize_experiment_heatmap_frame_accepts_optional_time_metadata() -> None:
    frame = normalize_experiment_heatmap_frame(
        _vector_df().drop(columns=["time_selected_h"]),
        experiment_id="experiment-2",
    )

    assert "time_selected_h" not in frame.columns
    assert frame["row_label"].tolist() == [
        "experiment-2::design-2",
        "experiment-2::design-10",
    ]


def test_normalize_experiment_heatmap_frame_accepts_nullable_time_metadata() -> None:
    vector = _vector_df()
    vector.loc[0, "time_selected_h"] = float("nan")

    frame = normalize_experiment_heatmap_frame(vector, experiment_id="experiment-2")

    assert pd.isna(frame.loc[0, "time_selected_h"])
    assert frame.loc[1, "time_selected_h"] == pytest.approx(12.0)


def test_four_state_vector_heatmap_runtime_persists_plot_bundle_record(tmp_path: Path) -> None:
    runtime = builtin_runtime()
    outputs = tmp_path / "outputs"
    store = runtime.record_store(outputs, plots_subdir="plots", exports_subdir="exports")
    store.persist_dataframe(
        producer_id="four_state_vector",
        producer_plugin="transform/four_state_vector",
        out_name="vector",
        record_id="four_state_vector/vector",
        df=_vector_df(),
        contract_id="logic.four_state_vector.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    decl = WorkbenchDecl(
        experiment=ExperimentDecl(id="vector-plot", title="vector-plot", lifecycle="active", root=tmp_path),
        experiment_semantics=ExperimentSemantics(
            protocol=ProtocolBinding(id="workbench/generic"),
            protocol_program=ProtocolSemanticProgram(protocol="workbench/generic"),
            annotations=AnnotationSemantics(),
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
                    id="four_state_vector_heatmap",
                    plugin="plot/four_state_vector_heatmap",
                    reads={"vector": RecordInputDecl(record_id="four_state_vector/vector")},
                    with_={"experiment_id": "vector-plot", "format": ["png"]},
                ),
            )
        ),
        exports=SurfaceDecl(specs=()),
    )

    run_spec(
        decl,
        include_pipeline=False,
        include_plots=True,
        include_exports=False,
        plot_specs=resolve_workbench(decl).plots,
        log_level="ERROR",
        runtime=runtime,
    )

    latest_ids = {record.record_id for record in store.iter_latest_records()}
    assert "plot:four_state_vector_heatmap" in latest_ids
    assert (outputs / "plots" / "four_state_vector_heatmap.png").exists()
