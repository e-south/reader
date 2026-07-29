from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reader.contracts import builtin_contract_catalog
from reader.domains.logic.sfxi.vec8_heatmap import normalize_experiment_vec8_heatmap_frame
from reader.errors import SFXIError
from reader.plugins.plot.sfxi_vec8_heatmap import SFXIVec8HeatmapCfg, SFXIVec8HeatmapPlot
from reader.protocols import ProtocolBinding, ProtocolSemanticProgram
from reader.runtime import builtin_runtime
from reader.workbench import PluginSemantics
from reader.workbench.assets import build_plugin_asset
from reader.workbench.decl.model import (
    ExperimentDecl,
    NotebookDecl,
    PipelineDecl,
    PluginStepDecl,
    RecordInputDecl,
    SurfaceDecl,
    WorkbenchDecl,
)
from reader.workbench.engine import run_spec
from reader.workbench.experiment import AnnotationSemantics, ExperimentSemantics, OutputLayout, ResourceCatalog
from reader.workbench.graph import resolve_workbench


def _vec8_df() -> pd.DataFrame:
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


def test_sfxi_vec8_heatmap_plot_saves_artifact(tmp_path: Path) -> None:
    ctx = SimpleNamespace(plots_dir=tmp_path, exp_dir=tmp_path / "experiment-2")
    cfg = SFXIVec8HeatmapCfg(experiment_id="experiment-2", format=["png"], title="SFXI vec8 heatmap")
    plugin = SFXIVec8HeatmapPlot()
    plugin.bind_runtime(
        descriptor=build_plugin_asset(
            plugin_id="plot/sfxi_vec8_heatmap",
            semantics=PluginSemantics(
                domain="logic",
                family="sfxi_vec8_heatmap",
                summary="SFXI vec8 heatmap plot.",
            ),
            plugin_cls=SFXIVec8HeatmapPlot,
        ),
        contracts=builtin_contract_catalog(),
    )

    output = plugin.run(ctx, {"vec8": _vec8_df()}, cfg)

    assert output["artifacts"] == [str(tmp_path / "sfxi_vec8_heatmap.png")]
    assert (tmp_path / "sfxi_vec8_heatmap.png").exists()


def test_normalize_experiment_vec8_heatmap_frame_rejects_missing_intensity_delta() -> None:
    with pytest.raises(SFXIError, match="requires column 'intensity_log2_offset_delta'"):
        normalize_experiment_vec8_heatmap_frame(
            _vec8_df().drop(columns=["intensity_log2_offset_delta"]),
            experiment_id="experiment-2",
        )


def test_normalize_experiment_vec8_heatmap_frame_accepts_optional_time_metadata() -> None:
    frame = normalize_experiment_vec8_heatmap_frame(
        _vec8_df().drop(columns=["time_selected_h"]),
        experiment_id="experiment-2",
    )

    assert "time_selected_h" not in frame.columns
    assert frame["row_label"].tolist() == [
        "experiment-2::design-2",
        "experiment-2::design-10",
    ]


def test_normalize_experiment_vec8_heatmap_frame_accepts_nullable_time_metadata() -> None:
    vec8 = _vec8_df()
    vec8.loc[0, "time_selected_h"] = float("nan")

    frame = normalize_experiment_vec8_heatmap_frame(vec8, experiment_id="experiment-2")

    assert pd.isna(frame.loc[0, "time_selected_h"])
    assert frame.loc[1, "time_selected_h"] == pytest.approx(12.0)


def test_sfxi_vec8_heatmap_runtime_persists_plot_bundle_record(tmp_path: Path) -> None:
    runtime = builtin_runtime()
    outputs = tmp_path / "outputs"
    store = runtime.record_store(outputs, plots_subdir="plots", exports_subdir="exports")
    store.persist_dataframe(
        producer_id="sfxi_vec8",
        producer_plugin="transform/sfxi",
        out_name="vec8",
        record_id="sfxi_vec8/vec8",
        df=_vec8_df(),
        contract_id="sfxi.vec8.v3",
        inputs=[],
        config_digest="sha256:test",
    )
    decl = WorkbenchDecl(
        experiment=ExperimentDecl(id="vec8-plot", title="vec8-plot", lifecycle="active", root=tmp_path),
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
                    id="sfxi_vec8_heatmap",
                    plugin="plot/sfxi_vec8_heatmap",
                    reads={"vec8": RecordInputDecl(record_id="sfxi_vec8/vec8")},
                    with_={"experiment_id": "vec8-plot", "format": ["png"]},
                ),
            )
        ),
        exports=SurfaceDecl(specs=()),
        notebooks=NotebookDecl(specs=()),
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
    assert "plot:sfxi_vec8_heatmap" in latest_ids
    assert (outputs / "plots" / "sfxi_vec8_heatmap.png").exists()
