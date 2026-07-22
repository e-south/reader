from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from reader.contracts import builtin_contract_catalog
from reader.plugins.plot.sfxi_setpoint_scatter import SFXISetpointScatterCfg, SFXISetpointScatterPlot
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
            "design_id": ["p01", "p02"],
            "reference_design_id": ["REF", "REF"],
            "time_selected_h": [10.0, 10.0],
            "intensity_log2_offset_delta": [0.0, 0.0],
            "r_logic": [8.0, 4.0],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 1.0],
            "y00_star": [0.0, 0.0],
            "y10_star": [0.0, 0.0],
            "y01_star": [0.0, 0.0],
            "y11_star": [1.0, 0.0],
            "flat_logic": [False, False],
        }
    )


def _install_fake_dnadesign_api(monkeypatch) -> None:
    class _FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeResult:
        api_version = "1"
        objective_name = "sfxi_v1"

        def to_records(self):
            return [
                {
                    "objective_name": "sfxi_v1",
                    "api_version": "1",
                    "state_order": ["00", "10", "01", "11"],
                    "setpoint_vector": [0.0, 0.0, 0.0, 1.0],
                    "denom_percentile": 95,
                    "denom_used": 2.0,
                    "logic_fidelity": 1.0,
                    "effect_raw": 2.0,
                    "effect_scaled": 1.0,
                    "sfxi": 1.0,
                    "clip_lo_mask": False,
                    "clip_hi_mask": True,
                    "intensity_disabled": False,
                },
                {
                    "objective_name": "sfxi_v1",
                    "api_version": "1",
                    "state_order": ["00", "10", "01", "11"],
                    "setpoint_vector": [0.0, 0.0, 0.0, 1.0],
                    "denom_percentile": 95,
                    "denom_used": 2.0,
                    "logic_fidelity": 1.0,
                    "effect_raw": 1.0,
                    "effect_scaled": 0.5,
                    "sfxi": 0.5,
                    "clip_lo_mask": False,
                    "clip_hi_mask": False,
                    "intensity_disabled": False,
                },
            ]

    fake_api = SimpleNamespace(
        SFXI_API_VERSION="1",
        SFXIScoringConfig=_FakeConfig,
        score_vec8=lambda *args, **kwargs: _FakeResult(),
    )
    real_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None):
        if name == "dnadesign.opal.api.sfxi":
            return fake_api
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)


def test_sfxi_setpoint_scatter_plot_saves_artifact(tmp_path, monkeypatch) -> None:
    _install_fake_dnadesign_api(monkeypatch)
    ctx = SimpleNamespace(plots_dir=tmp_path, palette_book=None)
    cfg = SFXISetpointScatterCfg(setpoints={"and": [0.0, 0.0, 0.0, 1.0]}, scaling_min_n=1)
    plugin = SFXISetpointScatterPlot()
    plugin.bind_runtime(
        descriptor=build_plugin_asset(
            plugin_id="plot/sfxi_setpoint_scatter",
            semantics=PluginSemantics(
                domain="logic",
                family="sfxi_objective_scatter",
                summary="SFXI setpoint scatter plot.",
            ),
            plugin_cls=SFXISetpointScatterPlot,
        ),
        contracts=builtin_contract_catalog(),
    )

    output = plugin.run(ctx, {"vec8": _vec8_df()}, cfg)

    assert output["artifacts"] == [str(tmp_path / "sfxi_setpoint_scatter.pdf")]
    assert (tmp_path / "sfxi_setpoint_scatter.pdf").exists()


def test_sfxi_setpoint_scatter_runtime_persists_plot_bundle_record(tmp_path: Path, monkeypatch) -> None:
    _install_fake_dnadesign_api(monkeypatch)
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
        experiment=ExperimentDecl(id="exp_sfxi_plot", title="exp_sfxi_plot", lifecycle="active", root=tmp_path),
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
                    id="sfxi_setpoint_scatter",
                    plugin="plot/sfxi_setpoint_scatter",
                    reads={"vec8": RecordInputDecl(record_id="sfxi_vec8/vec8")},
                    with_={"setpoints": {"and": [0.0, 0.0, 0.0, 1.0]}, "scaling_min_n": 1},
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
    assert "plot:sfxi_setpoint_scatter" in latest_ids
    assert (outputs / "plots" / "sfxi_setpoint_scatter.pdf").exists()
