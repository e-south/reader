from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection
from typer.testing import CliRunner

from reader.domains.logic.sfxi import triptych_sequence, triptych_sequence_outputs
from reader.domains.promoter import sequence_panel
from reader.domains.promoter.candidate_bindings import load_promoter_candidate_bindings
from reader.domains.promoter.sequence_panel import PromoterSequencePanelError
from reader.errors import SFXIError
from reader.protocols import ProtocolBinding, ProtocolSemanticProgram
from reader.runtime import builtin_runtime
from reader.tests.domains.plate_reader.analysis.response_window.test_promoter_evidence_bindings import (
    _write_binding_fixture,
)
from reader.tests.support import base_reader_config, cli_success_data, write_config
from reader.workbench.cli import app
from reader.workbench.decl.model import (
    ExperimentDecl,
    NotebookDecl,
    PipelineDecl,
    PluginStepDecl,
    RecordInputDecl,
    ResourceInputDecl,
    SurfaceDecl,
    WorkbenchDecl,
)
from reader.workbench.engine import run_spec
from reader.workbench.experiment import (
    AnnotationSemantics,
    ExperimentSemantics,
    OrderedStateSpaces,
    OrderedStateSpaceSpec,
    OutputLayout,
    ResourceCatalog,
    ResourceEntry,
)
from reader.workbench.graph import resolve_workbench

_TREATMENT_MAP = {
    "00": "EtOH 0%, 0 nM cipro",
    "10": "EtOH 3%, 0 nM cipro",
    "01": "EtOH 0%, 100 nM cipro",
    "11": "EtOH 3%, 100 nM cipro",
}


def _vec8_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": ["pDual-10-test01"],
            "id": ["seq01"],
            "sequence": ["ACGTACGT"],
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
    treatments = list(_TREATMENT_MAP.values())
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


def _normalized_render_config(**overrides: object) -> dict[str, object]:
    return triptych_sequence._normalize_config(
        {
            "state_map_ref": "induction_logic",
            "treatment_column": "treatment_alias",
            "treatment_map": _TREATMENT_MAP,
            "treatment_case_sensitive": True,
            **overrides,
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
    sequence_length_bp: int = 8
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


def _install_fake_sequence_panel(monkeypatch) -> None:
    monkeypatch.setattr(triptych_sequence, "require_baserender_api", lambda: _FakeBaseRender)
    monkeypatch.setattr(sequence_panel, "require_baserender_api", lambda: _FakeBaseRender)


def _triptych_publish_paths(
    tmp_path: Path, *, movie_enabled: bool = True
) -> tuple[dict[str, Path], dict[str, Path], Path]:
    outputs_dir = tmp_path / "outputs"
    ctx = SimpleNamespace(
        outputs_dir=outputs_dir,
        plots_dir=outputs_dir / "plots",
        exports_dir=outputs_dir / "exports",
    )
    final = triptych_sequence_outputs.bundle_paths(ctx=ctx, bundle_id="triptych")
    staging_root = outputs_dir / ".staging" / "triptych__test"
    staging_root.mkdir(parents=True)
    staging = triptych_sequence_outputs.staging_paths(
        staging_root=staging_root,
        bundle_id="triptych",
        movie_enabled=movie_enabled,
    )
    return staging, final, staging_root


def test_sfxi_triptych_relative_path_rejects_artifact_outside_outputs(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    with pytest.raises(ValueError, match="outside the declared outputs directory"):
        triptych_sequence_outputs.relative_to_outputs(tmp_path / "outside.pdf", outputs_dir=outputs)


def _write_triptych_bundle(paths: dict[str, Path], *, content: str) -> None:
    for key in ("poster", "pdf", "index", "manifest", "movie"):
        if key not in paths:
            continue
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_text(f"{content}:{key}\n", encoding="utf-8")
    paths["frames_dir"].mkdir(parents=True, exist_ok=True)
    (paths["frames_dir"] / "001_design.png").write_text(f"{content}:frame\n", encoding="utf-8")


def _assert_triptych_bundle(paths: dict[str, Path], *, content: str, include_movie: bool = True) -> None:
    for key in ("poster", "pdf", "index", "manifest", "movie"):
        if key not in paths or (key == "movie" and not include_movie):
            continue
        assert paths[key].read_text(encoding="utf-8") == f"{content}:{key}\n"
    assert (paths["frames_dir"] / "001_design.png").read_text(encoding="utf-8") == f"{content}:frame\n"


def _assert_no_triptych_transaction_paths(root: Path) -> None:
    names = {path.name for path in root.rglob("*")}
    assert not {name for name in names if ".backup" in name or ".__previous" in name}


def test_logic_sfxi_plot_list_surfaces_triptych_sequence(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_logic_triptych",
        protocol_id="logic/sfxi_screen",
        protocol_inputs={"state_map_ref": "induction_logic"},
        protocol_analysis={
            "include_vec8": True,
            "include_fold_change": False,
            "sfxi_triptych_sequence": {"candidate_bindings_resource": "promoter_candidate_bindings"},
        },
        protocol_outputs={"plots": {"profile": "none", "include": ["sfxi_triptych_sequence"]}},
        resources={
            "sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"},
            "promoter_candidate_bindings": {
                "kind": "file",
                "path": "./inputs/promoter_candidate_bindings/manifest.json",
            },
        },
        annotations={
            "ordered_state_spaces": {
                "induction_logic": {
                    "column": "treatment",
                    "state_order": ["00", "10", "01", "11"],
                    "values": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )
    cfg_path = write_config(tmp_path, cfg)
    result = CliRunner().invoke(app, ["plot", str(cfg_path), "--list", "--format", "json"])

    assert result.exit_code == 0
    payload = cli_success_data(result.output)
    assert payload["summary"]["by_plugin"] == {"plot/sfxi_triptych_sequence": 1}
    assert payload["plots"][0]["id"] == "sfxi_triptych_sequence"
    reads = {item["label"]: item for item in payload["plots"][0]["reads"]}
    assert reads["vec8"]["ref"] == {"record": "sfxi_vec8/vec8"}
    assert reads["assay"]["ref"] == {"record": "promote_to_tidy_plus_map/df"}
    assert reads["candidate_bindings"]["ref"]["resource"] == "promoter_candidate_bindings"


def test_sfxi_triptych_sequence_dependency_check_wraps_transitive_import_failures(monkeypatch) -> None:
    def fake_import_module(name: str):
        assert name == "dnadesign.baserender"
        raise ModuleNotFoundError("No module named 'Bio'")

    monkeypatch.setattr(sequence_panel.importlib, "import_module", fake_import_module)

    with pytest.raises(PromoterSequencePanelError, match="public dnadesign BaseRender API") as exc_info:
        sequence_panel.require_baserender_api()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_sfxi_triptych_requires_an_exact_design_binding(tmp_path: Path) -> None:
    bindings = load_promoter_candidate_bindings(_write_binding_fixture(tmp_path, reader_design_id="different-design"))

    with pytest.raises(SFXIError, match="absent from the exact candidate-binding artifact"):
        triptych_sequence._build_candidate_plan(
            vec8=_vec8_df(),
            bindings=bindings,
            cfg=_normalized_render_config(),
        )


def test_sfxi_triptych_rejects_reader_sequence_drift_from_binding(tmp_path: Path) -> None:
    bindings = load_promoter_candidate_bindings(_write_binding_fixture(tmp_path, reader_design_id="pDual-10-test01"))
    vec8 = _vec8_df()
    vec8.loc[0, "sequence"] = "TTTTTTTT"

    with pytest.raises(SFXIError, match="sequence disagrees"):
        triptych_sequence._build_candidate_plan(
            vec8=vec8,
            bindings=bindings,
            cfg=_normalized_render_config(),
        )


@pytest.mark.parametrize("missing_sequence", [None, pd.NA, np.nan])
def test_sfxi_triptych_uses_binding_when_optional_vec8_sequence_is_missing(
    tmp_path: Path,
    missing_sequence: object,
) -> None:
    bindings = load_promoter_candidate_bindings(_write_binding_fixture(tmp_path, reader_design_id="pDual-10-test01"))
    vec8 = _vec8_df()
    vec8.loc[0, "sequence"] = missing_sequence

    plan = triptych_sequence._build_candidate_plan(
        vec8=vec8,
        bindings=bindings,
        cfg=_normalized_render_config(),
    )

    assert plan.loc[0, "candidate_binding"].canonical_sequence == "ACGTACGT"


def test_sfxi_triptych_requires_resolved_state_map_treatment_semantics() -> None:
    with pytest.raises(SFXIError, match="state_map_ref must be a non-empty string"):
        triptych_sequence._normalize_config({})


@pytest.mark.parametrize("limit", [0, -1])
def test_sfxi_triptych_rejects_nonpositive_render_limits(limit: int) -> None:
    with pytest.raises(
        SFXIError,
        match=r"analysis\.sfxi_triptych_sequence\.limit must be a positive integer",
    ):
        _normalized_render_config(limit=limit)


def test_sfxi_triptych_maps_authored_labels_to_stable_states() -> None:
    cfg = _normalized_render_config()

    mapped = triptych_sequence._bind_treatment_states(assay=_assay_df(), cfg=cfg)

    assert set(mapped[cfg["state_column"]].dropna()) == {"00", "10", "01", "11"}


def test_sfxi_triptych_honors_state_map_case_sensitivity() -> None:
    lower_labels = {state: label.lower() for state, label in _TREATMENT_MAP.items()}
    cfg = _normalized_render_config(treatment_case_sensitive=False, treatment_map=lower_labels)

    mapped = triptych_sequence._bind_treatment_states(assay=_assay_df(), cfg=cfg)

    assert set(mapped[cfg["state_column"]].dropna()) == {"00", "10", "01", "11"}


def test_sfxi_triptych_snapshot_shows_observed_wells_without_bars() -> None:
    cfg = _normalized_render_config()
    mapped = triptych_sequence._bind_treatment_states(assay=_assay_df(), cfg=cfg)
    mapped["plot_time_h"] = mapped["time"]
    figure, axis = plt.subplots()

    try:
        triptych_sequence._draw_snapshot_panel(
            axis,
            assay=mapped,
            channel="YFP/CFP",
            snapshot_time_h=12.0,
            tolerance_h=0.01,
            states=list(cfg["states"]),
            cfg=cfg,
            y_limits={"YFP/CFP": [0.0, 3.0]},
        )

        assert not axis.patches
        replicate_clouds = [
            item
            for item in axis.collections
            if isinstance(item, PathCollection)
            and len(item.get_offsets()) == 2
            and len(item.get_edgecolors())
            and mcolors.to_hex(item.get_edgecolors()[0]) == "#94a3b8"
        ]
        assert len(replicate_clouds) == 4
        assert sum(len(item.get_offsets()) for item in replicate_clouds) == 8
        assert all(mcolors.to_hex(item.get_facecolors()[0]) == "#ffffff" for item in replicate_clouds)
        summary_values = [
            float(item.get_segments()[0][0, 1])
            for item in axis.collections
            if isinstance(item, LineCollection)
            and len(item.get_segments()) == 1
            and np.isclose(abs(np.diff(item.get_segments()[0][:, 0])[0]), 0.36)
        ]
        expected = (
            mapped.loc[mapped["channel"].eq("YFP/CFP") & mapped["time"].eq(12.0)]
            .groupby(cfg["state_column"])["value"]
            .mean()
            .reindex(("00", "10", "01", "11"))
            .to_numpy(dtype=float)
        )
        np.testing.assert_allclose(summary_values, expected)
    finally:
        plt.close(figure)


def test_sfxi_triptych_global_scale_contains_snapshot_sd_whiskers() -> None:
    cfg = _normalized_render_config()
    mapped = triptych_sequence._bind_treatment_states(assay=_assay_df(), cfg=cfg)
    mask = mapped["channel"].eq("YFP/CFP") & mapped["time"].eq(12.0) & mapped[cfg["state_column"]].eq("00")
    mapped.loc[mask, "value"] = [0.0, 100.0]
    scales = triptych_sequence._compute_render_scales(
        assay=mapped,
        render_plan=pd.DataFrame({cfg["design_col"]: ["pDual-10-test01"]}),
        cfg=cfg,
    )

    whisker_upper = 50.0 + np.std([0.0, 100.0], ddof=1)
    assert scales["y_limits"]["YFP/CFP"][1] >= whisker_upper


def test_sfxi_triptych_rejects_duplicated_treatment_identity() -> None:
    with pytest.raises(SFXIError, match="comes from the resolved ordered state-space contract"):
        _normalized_render_config(treatments=[])


def test_sfxi_triptych_frame_filename_keeps_colliding_label_slugs_unique() -> None:
    first = triptych_sequence._frame_filename(row_number=1, display_label="A-1")
    second = triptych_sequence._frame_filename(row_number=2, display_label="a_1")

    assert first == "001_a_1.png"
    assert second == "002_a_1.png"
    assert first != second


def test_sfxi_triptych_publish_removes_partial_new_bundle_after_mid_commit_failure(tmp_path: Path, monkeypatch) -> None:
    staging, final, staging_root = _triptych_publish_paths(tmp_path)
    _write_triptych_bundle(staging, content="new")
    original_replace = Path.replace

    def fail_on_manifest_install(path: Path, target: Path):
        if path == staging["manifest"]:
            raise OSError("injected manifest install failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_on_manifest_install)

    with pytest.raises(OSError, match="injected manifest install failure"):
        triptych_sequence_outputs.publish_bundle(staging=staging, final=final)

    assert not any(path.exists() for path in final.values())
    assert not staging_root.exists()


def test_sfxi_triptych_publish_restores_complete_bundle_after_mid_commit_failure(tmp_path: Path, monkeypatch) -> None:
    staging, final, staging_root = _triptych_publish_paths(tmp_path)
    _write_triptych_bundle(final, content="old")
    _write_triptych_bundle(staging, content="new")
    original_replace = Path.replace

    def fail_on_manifest_install(path: Path, target: Path):
        if path == staging["manifest"]:
            raise OSError("injected manifest install failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_on_manifest_install)

    with pytest.raises(OSError, match="injected manifest install failure"):
        triptych_sequence_outputs.publish_bundle(staging=staging, final=final)

    _assert_triptych_bundle(final, content="old")
    assert not staging_root.exists()
    _assert_no_triptych_transaction_paths(tmp_path)


def test_sfxi_triptych_publish_restores_backups_when_backup_stage_fails(tmp_path: Path, monkeypatch) -> None:
    staging, final, staging_root = _triptych_publish_paths(tmp_path)
    _write_triptych_bundle(final, content="old")
    _write_triptych_bundle(staging, content="new")
    original_replace = Path.replace

    def fail_on_manifest_backup(path: Path, target: Path):
        if path == final["manifest"]:
            raise OSError("injected manifest backup failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_on_manifest_backup)

    with pytest.raises(OSError, match="injected manifest backup failure"):
        triptych_sequence_outputs.publish_bundle(staging=staging, final=final)

    _assert_triptych_bundle(final, content="old")
    assert not staging_root.exists()
    _assert_no_triptych_transaction_paths(tmp_path)


def test_sfxi_triptych_publish_rejects_incomplete_stage_before_touching_existing_bundle(tmp_path: Path) -> None:
    staging, final, staging_root = _triptych_publish_paths(tmp_path)
    _write_triptych_bundle(final, content="old")
    _write_triptych_bundle(staging, content="new")
    staging["index"].unlink()

    with pytest.raises(SFXIError, match="staged index is missing"):
        triptych_sequence_outputs.publish_bundle(staging=staging, final=final)

    _assert_triptych_bundle(final, content="old")
    assert not staging_root.exists()
    _assert_no_triptych_transaction_paths(tmp_path)


def test_sfxi_triptych_publish_replaces_complete_bundle_and_cleans_transaction_paths(tmp_path: Path) -> None:
    staging, final, staging_root = _triptych_publish_paths(tmp_path)
    _write_triptych_bundle(final, content="old")
    _write_triptych_bundle(staging, content="new")

    triptych_sequence_outputs.publish_bundle(staging=staging, final=final)

    _assert_triptych_bundle(final, content="new")
    assert not staging_root.exists()
    _assert_no_triptych_transaction_paths(tmp_path)


def test_sfxi_triptych_publish_reports_post_commit_cleanup_failure_without_failing_recording(
    tmp_path: Path, monkeypatch
) -> None:
    staging, final, staging_root = _triptych_publish_paths(tmp_path)
    _write_triptych_bundle(final, content="old")
    _write_triptych_bundle(staging, content="new")

    def fail_cleanup(path: Path) -> None:
        assert path == staging_root
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(triptych_sequence_outputs, "cleanup_staging_root", fail_cleanup)

    with pytest.warns(RuntimeWarning, match="publication committed, but transaction cleanup failed"):
        triptych_sequence_outputs.publish_bundle(staging=staging, final=final)

    _assert_triptych_bundle(final, content="new")
    assert staging_root.exists()


def test_sfxi_triptych_publish_removes_stale_movie_when_movie_is_disabled(tmp_path: Path) -> None:
    staging, final, staging_root = _triptych_publish_paths(tmp_path, movie_enabled=False)
    _write_triptych_bundle(final, content="old")
    _write_triptych_bundle(staging, content="new")

    triptych_sequence_outputs.publish_bundle(staging=staging, final=final)

    _assert_triptych_bundle(final, content="new", include_movie=False)
    assert not final["movie"].exists()
    assert not staging_root.exists()
    _assert_no_triptych_transaction_paths(tmp_path)


def test_sfxi_triptych_publish_restores_stale_movie_after_disabled_movie_install_failure(
    tmp_path: Path, monkeypatch
) -> None:
    staging, final, staging_root = _triptych_publish_paths(tmp_path, movie_enabled=False)
    _write_triptych_bundle(final, content="old")
    _write_triptych_bundle(staging, content="new")
    original_replace = Path.replace

    def fail_on_manifest_install(path: Path, target: Path):
        if path == staging["manifest"]:
            raise OSError("injected manifest install failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_on_manifest_install)

    with pytest.raises(OSError, match="injected manifest install failure"):
        triptych_sequence_outputs.publish_bundle(staging=staging, final=final)

    _assert_triptych_bundle(final, content="old")
    assert final["movie"].read_text(encoding="utf-8") == "old:movie\n"
    assert not staging_root.exists()
    _assert_no_triptych_transaction_paths(tmp_path)


def test_sfxi_triptych_sequence_runtime_persists_bundle_record(tmp_path: Path, monkeypatch) -> None:
    _install_fake_sequence_panel(monkeypatch)
    binding_root = _write_binding_fixture(tmp_path, reader_design_id="pDual-10-test01")
    binding_manifest = binding_root / "manifest.json"
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
            protocol=ProtocolBinding(id="logic/sfxi_screen"),
            protocol_program=ProtocolSemanticProgram(protocol="logic/sfxi_screen"),
            annotations=AnnotationSemantics(
                ordered_state_spaces=OrderedStateSpaces(
                    by_id={
                        "induction_logic": OrderedStateSpaceSpec(
                            column="treatment_alias",
                            state_order=("00", "10", "01", "11"),
                            source_values=_TREATMENT_MAP,
                            case_sensitive=True,
                        )
                    }
                )
            ),
            resources=ResourceCatalog(
                by_id={"promoter_candidate_bindings": ResourceEntry(kind="file", path=binding_manifest)}
            ),
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
                        "candidate_bindings": ResourceInputDecl(resource_id="promoter_candidate_bindings"),
                    },
                    with_={
                        "acquisition_transition_time_h": 12.0,
                        "state_map_ref": "induction_logic",
                    },
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
    assert manifest["schema"] == "reader.sfxi_triptych_sequence_bundle.v3"
    assert manifest["candidate_bindings"]["manifest_sha256"].startswith("sha256:")
    assert manifest["treatment_contract"] == {
        "state_map_ref": "induction_logic",
        "column": "treatment_alias",
        "corners": _TREATMENT_MAP,
        "case_sensitive": True,
    }
    assert manifest["row_order"] == ["pDual-10-test01"]
    record = manifest["records"][0]
    assert record["candidate_id"] == "candidate-spyp"
    assert record["sequence_authority_dataset_id"] == "source-dataset"
    assert record["acquisition_transition_time_h"] == 12.0
    assert "induction_time_h" not in record
    assert ".staging" not in record["png_path"]
    assert (outputs / record["png_path"]).exists()
    index = pd.read_csv(outputs / "exports" / "sfxi_triptych_sequence" / "sfxi_triptych_sequence_index.csv")
    assert index["png_path"].tolist() == [record["png_path"]]
    assert (outputs / index.loc[0, "png_path"]).exists()
