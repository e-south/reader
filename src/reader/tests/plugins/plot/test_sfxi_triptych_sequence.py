from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from reader.domains.logic.sfxi import triptych_sequence, triptych_sequence_dnadesign
from reader.errors import SFXIError
from reader.protocols import ProtocolBinding, ProtocolSemanticProgram
from reader.runtime import builtin_runtime
from reader.tests.support import base_reader_config, write_config
from reader.workbench.cli import app
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
            "design_id": ["pDual-10-test01"],
            "id": ["seq01"],
            "sequence": ["ACGTACGTACGT"],
            "reference_design_id": ["pDual-10"],
            "intensity_log2_offset_delta": [0.0],
            "time_selected_h": [12.0],
            "v00": [0.0],
            "v10": [1.0],
            "v01": [0.0],
            "v11": [0.0],
            "y00_star": [0.0],
            "y10_star": [1.0],
            "y01_star": [0.0],
            "y11_star": [0.0],
            "r_logic": [4.0],
            "flat_logic": [False],
        }
    )


def _assay_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    treatments = ["negative", "3% EtOH", "100 nM ciprofloxacin", "3% EtOH + 100 nM ciprofloxacin"]
    for treatment_idx, treatment in enumerate(treatments, start=1):
        for time in (0.0, 12.0, 24.0):
            for rep in (1, 2):
                rows.append(
                    {
                        "design_id": "pDual-10-test01",
                        "position": f"A{treatment_idx}{rep}",
                        "time": time,
                        "channel": "OD600",
                        "value": 0.12 + 0.018 * time + 0.01 * rep,
                        "treatment": treatment,
                        "treatment_alias": treatment,
                    }
                )
                rows.append(
                    {
                        "design_id": "pDual-10-test01",
                        "position": f"B{treatment_idx}{rep}",
                        "time": time,
                        "channel": "YFP/CFP",
                        "value": 1.0 + 0.05 * treatment_idx + 0.02 * time + 0.01 * rep,
                        "treatment": treatment,
                        "treatment_alias": treatment,
                    }
                )
    return pd.DataFrame(rows)


def _sequence_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "usr_sequence_id": ["seq01"],
            "usr_sequence": ["ACGTACGTACGT"],
            "usr_label": ["pDual-10-test01"],
            "usr_annotations": [[]],
            "usr_dataset": ["usr_test_promoters"],
            "sequence_adapter_kind": ["densegen_tfbs"],
        }
    )


@dataclass(frozen=True)
class _FakeDiagnostics:
    contract_id: str = "dnadesign.baserender.sequence_panel.v1"
    contract_version: str = "1"
    style_profile: str = "promoter_compact_slide.v1"
    style_preset: str = "presentation_default"
    adapter_kind: str = "densegen_tfbs"
    renderer_name: str = "sequence_rows"
    sequence_length_bp: int = 12
    feature_count: int = 1
    strand_count: int = 2
    legend_entries: tuple[str, ...] = ("promoter",)
    image_width_px: int = 220
    image_height_px: int = 60


class _FakeBaseRender:
    BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION = "1"

    @staticmethod
    def render_sequence_panel_image(*args, **kwargs):
        del args, kwargs
        image = np.full((60, 220, 4), 255, dtype=np.uint8)
        image[15:45, 20:200, :3] = 100
        return SimpleNamespace(image=image, diagnostics=_FakeDiagnostics())

    @staticmethod
    def sequence_panel_config_for_adapter(*args, **kwargs):
        del args, kwargs
        return object()


class _FakeUsrWithTransitiveImportFailure:
    def __getattr__(self, name: str):
        if name == "Dataset":
            raise ModuleNotFoundError("No module named 'Bio'")
        raise AttributeError(name)


def _install_fake_sequence_panel(monkeypatch) -> None:
    monkeypatch.setattr(triptych_sequence, "require_dnadesign_sequence_panel_api", lambda: (_FakeBaseRender, object()))
    monkeypatch.setattr(triptych_sequence, "_load_usr_rows", lambda *, usr, cfg, exp_dir: _sequence_rows())


def test_logic_sfxi_plot_list_surfaces_triptych_sequence(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_logic_triptych",
        protocol_id="logic/sfxi_screen",
        protocol_inputs={"logic_map_ref": "induction_logic"},
        protocol_analysis={
            "include_vec8": True,
            "include_fold_change": False,
            "sfxi_triptych_sequence": {"sequence_source": {"dataset": "usr_test_promoters"}},
        },
        protocol_outputs={"plots": {"profile": "none", "include": ["sfxi_triptych_sequence"]}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "logic_maps": {
                "induction_logic": {
                    "column": "treatment",
                    "corners": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    cfg_path = write_config(tmp_path, cfg)
    result = CliRunner().invoke(app, ["plot", str(cfg_path), "--list", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["by_plugin"] == {"plot/sfxi_triptych_sequence": 1}
    assert payload["plots"][0]["id"] == "sfxi_triptych_sequence"
    reads = {item["label"]: item for item in payload["plots"][0]["reads"]}
    assert reads["vec8"]["ref"] == {"record": "sfxi_vec8/vec8"}
    assert reads["assay"]["ref"] == {"record": "promote_to_tidy_plus_map/df"}


def test_sfxi_triptych_sequence_dependency_check_wraps_transitive_import_failures(monkeypatch) -> None:
    def fake_import_module(name: str):
        if name == "dnadesign.baserender":
            return _FakeBaseRender
        if name == "dnadesign.usr":
            return _FakeUsrWithTransitiveImportFailure()
        raise AssertionError(name)

    monkeypatch.setattr(triptych_sequence_dnadesign.importlib, "import_module", fake_import_module)

    with pytest.raises(SFXIError, match="requires dnadesign public APIs") as exc_info:
        triptych_sequence_dnadesign.require_dnadesign_sequence_panel_api()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_sfxi_triptych_frame_filename_keeps_colliding_label_slugs_unique() -> None:
    first = triptych_sequence._frame_filename(row_number=1, display_label="A-1")
    second = triptych_sequence._frame_filename(row_number=2, display_label="a_1")

    assert first == "001_a_1.png"
    assert second == "002_a_1.png"
    assert first != second


def test_sfxi_triptych_sequence_runtime_persists_bundle_record(tmp_path: Path, monkeypatch) -> None:
    _install_fake_sequence_panel(monkeypatch)
    runtime = builtin_runtime()
    outputs = tmp_path / "outputs"
    store = runtime.record_store(outputs, plots_subdir="plots", exports_subdir="exports")
    store.persist_dataframe(
        producer_id="sfxi_vec8",
        producer_plugin="transform/sfxi",
        out_name="vec8",
        record_id="sfxi_vec8/vec8",
        df=_vec8_df(),
        contract_id="sfxi.vec8.v2",
        inputs=[],
        config_digest="sha256:test-vec8",
    )
    store.persist_dataframe(
        producer_id="promote_to_tidy_plus_map",
        producer_plugin="validator/to_tidy_plus_map",
        out_name="df",
        record_id="promote_to_tidy_plus_map/df",
        df=_assay_df(),
        contract_id="plate_reader.annotated.v1",
        inputs=[],
        config_digest="sha256:test-assay",
    )
    decl = WorkbenchDecl(
        experiment=ExperimentDecl(id="exp_sfxi_triptych", title="exp_sfxi_triptych", lifecycle="active", root=tmp_path),
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
                    id="sfxi_triptych_sequence",
                    plugin="plot/sfxi_triptych_sequence",
                    reads={
                        "vec8": RecordInputDecl(record_id="sfxi_vec8/vec8"),
                        "assay": RecordInputDecl(record_id="promote_to_tidy_plus_map/df"),
                    },
                    with_={"sequence_source": {"dataset": "usr_test_promoters"}},
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

    latest = {record.record_id: record for record in store.iter_latest_records()}
    assert "plot:sfxi_triptych_sequence" in latest
    assert (outputs / "plots" / "sfxi_triptych_sequence" / "sfxi_triptych_sequence.pdf").exists()
    assert (outputs / "plots" / "sfxi_triptych_sequence" / "sfxi_triptych_sequence.png").exists()
    manifest_path = outputs / "manifests" / "sfxi_triptych_sequence_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "reader.sfxi_triptych_sequence_bundle.v1"
    assert manifest["row_order"] == ["pDual-10-test01"]
    record = manifest["records"][0]
    assert ".staging" not in record["png_path"]
    assert (outputs / record["png_path"]).exists()
    index = pd.read_csv(outputs / "exports" / "sfxi_triptych_sequence" / "sfxi_triptych_sequence_index.csv")
    assert index["png_path"].tolist() == [record["png_path"]]
    assert (outputs / index.loc[0, "png_path"]).exists()
