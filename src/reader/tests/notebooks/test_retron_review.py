from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import yaml

from reader.contracts import builtin_contract_catalog
from reader.domains.plate_reader.plots.common import annotate_points_smart
from reader.tests.support import base_reader_config, write_config
from reader.workbench.notebooks.retron_review import (
    RetronReviewSource,
    build_architecture_frame,
    build_expected_vs_observed_frame,
    build_label_value_options,
    build_specificity_matrix,
    contextualize_retron_plot_copy,
    filter_supporting_table_for_figure,
    load_cached_parquet_frame,
    load_notebook_workbench_context,
    load_retron_review_bundle,
    load_retron_source_surface,
    render_retron_aggregate_plot,
    render_retron_aggregate_plot_cached,
    render_retron_experiment_plot,
    render_retron_experiment_plot_cached,
    retron_figure_coverage_rows,
    retron_plot_rendered_files,
    retron_source_surface_overview_rows,
    retron_table_kwargs,
)
from reader.workbench.records import RecordStore


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_retron_review_bundle_loads_manifest_and_builds_cross_run_frames(tmp_path: Path) -> None:
    mono_summary = _write_csv(
        tmp_path / "exports" / "mono_summary.csv",
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "S_AUC",
                "value": 0.40,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "spyP",
                "sponge": "BaeR",
                "metric": "S_AUC",
                "value": 0.60,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "sulAp",
                "sponge": "LexA",
                "metric": "S_AUC",
                "value": 0.50,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxR",
                "metric": "S_AUC",
                "value": 0.30,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxS",
                "metric": "S_AUC",
                "value": 0.20,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
        ],
    )
    mono_trace = _write_csv(tmp_path / "exports" / "mono_trace.csv", [{"metric": "D", "value": 0.1}])
    multi_summary = _write_csv(
        tmp_path / "exports" / "multi_summary.csv",
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR-LexA",
                "metric": "S_AUC",
                "value": 0.35,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "sulAp",
                "sponge": "CpxR-LexA",
                "metric": "S_AUC",
                "value": 0.45,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "spyP",
                "sponge": "BaeR-LexA-SoxR-SoxS",
                "metric": "S_AUC",
                "value": 0.55,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "quad",
            },
            {
                "sensor": "sulAp",
                "sponge": "BaeR-LexA-SoxR-SoxS",
                "metric": "S_AUC",
                "value": 0.52,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "quad",
            },
            {
                "sensor": "soxSp",
                "sponge": "BaeR-LexA-SoxR-SoxS",
                "metric": "S_AUC",
                "value": 0.48,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "quad",
            },
        ],
    )
    multi_trace = _write_csv(tmp_path / "exports" / "multi_trace.csv", [{"metric": "D", "value": 0.2}])
    manifest_path = tmp_path / "review_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "relevant_stress_map": {
                    "spyP": "3% EtOH",
                    "sulAp": "100 nM ciprofloxacin",
                    "soxSp": "15 µM PMS",
                },
                "sensor_target_map": {
                    "spyP": ["CpxR", "BaeR"],
                    "sulAp": ["LexA"],
                    "soxSp": ["SoxR", "SoxS"],
                },
                "sources": [
                    {
                        "label": "mono",
                        "experiment_id": "mono_family",
                        "summary": str(mono_summary.relative_to(tmp_path)),
                        "trace": str(mono_trace.relative_to(tmp_path)),
                    },
                    {
                        "label": "multi",
                        "experiment_id": "multi_family",
                        "summary": str(multi_summary.relative_to(tmp_path)),
                        "trace": str(multi_trace.relative_to(tmp_path)),
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    bundle = load_retron_review_bundle(manifest_path)
    specificity = build_specificity_matrix(bundle.summary_df, score_metric="S_AUC")
    architecture = build_architecture_frame(
        bundle.summary_df,
        sensor_target_map=bundle.sensor_target_map,
        score_metric="S_AUC",
    )
    expected_vs_observed = build_expected_vs_observed_frame(
        bundle.summary_df,
        sensor_target_map=bundle.sensor_target_map,
        score_metric="S_AUC",
    )

    assert {item.label for item in bundle.sources} == {"mono", "multi"}
    assert set(bundle.summary_df["source_experiment_id"]) == {"mono_family", "multi_family"}
    assert float(specificity.loc["soxSp", "BaeR-LexA-SoxR-SoxS"]) == pytest.approx(0.48)
    quad_spy = architecture[
        (architecture["sensor"] == "spyP") & (architecture["sponge"] == "BaeR-LexA-SoxR-SoxS")
    ].iloc[0]
    assert int(quad_spy["motif_count"]) == 4
    assert int(quad_spy["irrelevant_motif_count"]) == 3
    quad_sox = expected_vs_observed[
        (expected_vs_observed["sensor"] == "soxSp") & (expected_vs_observed["sponge"] == "BaeR-LexA-SoxR-SoxS")
    ].iloc[0]
    assert float(quad_sox["expected_best_single"]) == pytest.approx(0.30)
    assert float(quad_sox["expected_sum"]) == pytest.approx(0.50)
    assert float(quad_sox["observed"]) == pytest.approx(0.48)


def test_retron_review_bundle_fails_fast_when_source_exports_are_missing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "review_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "relevant_stress_map": {"spyP": "3% EtOH"},
                "sensor_target_map": {"spyP": ["CpxR", "BaeR"]},
                "sources": [
                    {
                        "label": "missing",
                        "summary": "./exports/summary.csv",
                        "trace": "./exports/trace.csv",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="source exports are missing"):
        load_retron_review_bundle(manifest_path)


def test_retron_review_bundle_resolves_experiment_sources_relative_to_source_root(tmp_path: Path) -> None:
    aggregate_root = tmp_path / "20260319_retron_review"
    manifest_path = aggregate_root / "inputs" / "review_manifest.yaml"
    source_root = tmp_path / "20260313_mono_functional_sponges"
    _write_csv(
        source_root / "outputs" / "exports" / "retron" / "semantic_summary.csv",
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "S_AUC",
                "value": 0.40,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            }
        ],
    )
    _write_csv(
        source_root / "outputs" / "exports" / "retron" / "semantic_trace.csv",
        [{"metric": "D", "value": 0.1}],
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "relevant_stress_map": {"spyP": "3% EtOH"},
                "sensor_target_map": {"spyP": ["CpxR", "BaeR"]},
                "sources": [{"label": "mono", "experiment": "../20260313_mono_functional_sponges"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    bundle = load_retron_review_bundle(manifest_path, source_root=aggregate_root)

    assert (
        bundle.sources[0].summary_path
        == (source_root / "outputs" / "exports" / "retron" / "semantic_summary.csv").resolve()
    )
    assert (
        bundle.sources[0].trace_path
        == (source_root / "outputs" / "exports" / "retron" / "semantic_trace.csv").resolve()
    )


def test_retron_figure_coverage_rows_call_out_cross_run_and_follow_on_views() -> None:
    coverage = retron_figure_coverage_rows()
    by_figure = {row["Figure"]: row for row in coverage}

    assert by_figure["Figure 13 — Specificity matrix"]["Coverage"] == "Exact aggregate notebook figure"
    assert by_figure["Figure 19 — Plate-position heatmaps"]["Coverage"] == "Not first-class compiled yet"


def test_load_notebook_workbench_context_uses_builtin_protocol_catalog(tmp_path: Path) -> None:
    cfg_path = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="exp_retron_context",
            protocol_id="plate_reader/retron_sponge_screen",
            protocol_analysis={
                "semantic_metrics": {
                    "relevant_stress_map": {"spyP": "3% EtOH"},
                    "sensor_target_map": {"spyP": ["CpxR", "BaeR"]},
                }
            },
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        ),
    )

    context = load_notebook_workbench_context(cfg_path)

    assert context.config_path == cfg_path
    assert context.decl.experiment_semantics.protocol.id == "plate_reader/retron_sponge_screen"
    assert context.workbench.plots


def test_load_retron_source_surface_reads_scoped_plot_catalog_and_record_paths(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "metadata.xlsx").write_text("stub", encoding="utf-8")
    cfg_path = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="exp_retron_source_surface",
            title="Source surface test",
            protocol_id="plate_reader/retron_sponge_screen",
            protocol_analysis={
                "semantic_metrics": {
                    "relevant_stress_map": {"spyP": "3% EtOH"},
                    "sensor_target_map": {"spyP": ["CpxR", "BaeR"]},
                }
            },
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        ),
    )
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    tidy = pd.DataFrame(
        [
            {
                "position": "A1",
                "time": 0.0,
                "channel": "OD600",
                "value": 0.1,
            }
        ]
    )
    trace = pd.DataFrame(
        [
            {
                "position": "A1",
                "time": 0.0,
                "channel": "trace",
                "value": -0.4,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "position": "A1",
                "time": 0.0,
                "channel": "summary",
                "value": 0.4,
            }
        ]
    )
    store.persist_dataframe(
        producer_id="ratio_yfp_od600",
        producer_plugin="transform/ratio_yfp_od600",
        out_name="df",
        record_id="ratio_yfp_od600/df",
        df=tidy,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:ratio",
    )
    store.persist_dataframe(
        producer_id="semantic_metrics",
        producer_plugin="transform/semantic_metrics",
        out_name="trace",
        record_id="semantic_metrics/trace",
        df=trace,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:trace",
    )
    store.persist_dataframe(
        producer_id="semantic_metrics",
        producer_plugin="transform/semantic_metrics",
        out_name="summary",
        record_id="semantic_metrics/summary",
        df=summary,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:summary",
    )

    source = RetronReviewSource(
        label="mono",
        experiment_id="exp_retron_source_surface",
        experiment_root=tmp_path,
        config_path=cfg_path,
        summary_path=cfg_path,
        trace_path=cfg_path,
    )

    surface = load_retron_source_surface(source)
    overview_rows = retron_source_surface_overview_rows(source, surface)

    assert surface.experiment_title == "Source surface test"
    assert surface.protocol_id == "plate_reader/retron_sponge_screen"
    assert any(row["Plot id"] == "raw_kinetics" for row in surface.plot_catalog_rows)
    assert any(record_id == "semantic_metrics/trace" for record_id, _ in surface.record_paths)
    assert any(record_id == "ratio_yfp_od600/df" for record_id, _ in surface.record_paths)
    assert overview_rows[0] == {"Field": "Source label", "Value": "mono"}
    assert overview_rows[3]["Field"] == "Compiled plots"


def test_retron_plot_rendered_files_accepts_legacy_raw_kinetics_prefix(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "ts_spyP_tetO.pdf").write_text("legacy", encoding="utf-8")

    matches = retron_plot_rendered_files(plots_dir, plot_id="raw_kinetics", plugin="plot/time_series")

    assert matches == ["ts_spyP_tetO.pdf"]


def test_render_retron_experiment_plot_reuses_canonical_trace_renderer() -> None:
    trace = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "tetO",
                "stress_condition": "H2O",
                "time_from_stress": 0.0,
                "metric": "C",
                "value": 0.0,
                "IPTG": "-IPTG",
                "relevant_sensor_pair": True,
            },
            {
                "sensor": "spyP",
                "sponge": "tetO",
                "stress_condition": "H2O",
                "time_from_stress": 1.0,
                "metric": "C",
                "value": 0.0,
                "IPTG": "+IPTG",
                "relevant_sensor_pair": True,
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "stress_condition": "H2O",
                "time_from_stress": 0.0,
                "metric": "C",
                "value": -0.2,
                "IPTG": "-IPTG",
                "relevant_sensor_pair": True,
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "stress_condition": "H2O",
                "time_from_stress": 1.0,
                "metric": "C",
                "value": -0.6,
                "IPTG": "+IPTG",
                "relevant_sensor_pair": True,
            },
        ]
    )
    plot_spec = {
        "id": "matched_control_kinetics",
        "plugin": "plot/retron_trace",
        "reads": {"trace": {"record": "semantic_metrics/trace"}},
        "with": {
            "metrics": ["C"],
            "title": "Matched-control-normalized kinetics",
            "filename": "matched_control_kinetics",
            "control_name": "tetO",
            "relevant_only": True,
            "stress_order": ["H2O"],
        },
    }

    result = render_retron_experiment_plot(plot_spec, datasets={"semantic_metrics/trace": trace})

    assert result.plot_id == "matched_control_kinetics"
    assert result.stage == "2. Assay kinetics"
    assert "same-sensor tetO subtraction" in result.question
    assert result.source_record == "semantic_metrics/trace"
    assert len(result.figures) == 1
    assert result.figures[0].filename == "matched_control_kinetics__sensor=spyp"
    assert result.figures[0].fig.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert set(result.supporting_table["metric"]) == {"C"}
    plt.close(result.figures[0].fig)


def test_contextualize_retron_plot_copy_replaces_generic_relevant_stress_text() -> None:
    supporting_table = pd.DataFrame(
        [
            {"sensor": "spyP", "stress_condition": "3% EtOH"},
            {"sensor": "spyP", "stress_condition": "3% EtOH"},
        ]
    )

    contextual = contextualize_retron_plot_copy(
        question="How does the sponge behave under relevant stress?",
        math="M_AUC=AUC(D(relevant stress)-D(H2O)).",
        meaning="Compares relevant stress against H2O for the selected sensor.",
        supporting_table=supporting_table,
        relevant_stress_map={"spyP": "3% EtOH"},
    )

    assert contextual["question"] == "How does the sponge behave under 3% EtOH?"
    assert contextual["math"] == "M_AUC=AUC(D(3% EtOH)-D(H2O))."
    assert contextual["meaning"] == "Compares 3% EtOH against H2O for the selected sensor."


def test_render_retron_experiment_plot_uses_descriptive_time_series_axis_labels() -> None:
    tidy = pd.DataFrame(
        [
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 0.0,
                "sheet_index": 0,
                "channel": "OD600",
                "value": 0.1,
            },
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 1.0,
                "sheet_index": 0,
                "channel": "OD600",
                "value": 0.2,
            },
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 0.0,
                "sheet_index": 0,
                "channel": "CFP",
                "value": 8000,
            },
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 1.0,
                "sheet_index": 0,
                "channel": "CFP",
                "value": 9000,
            },
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 0.0,
                "sheet_index": 0,
                "channel": "YFP",
                "value": 1200,
            },
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 1.0,
                "sheet_index": 0,
                "channel": "YFP",
                "value": 1800,
            },
        ]
    )
    plot_spec = {
        "id": "raw_kinetics",
        "plugin": "plot/time_series",
        "reads": {"df": {"record": "ratio_yfp_od600/df"}},
        "with": {
            "partition": {"by": "design_id"},
            "hue": "treatment",
            "xlabel": "Time from stress addition (h)",
            "y": ["OD600", "CFP", "YFP"],
            "ylabel_map": {
                "OD600": "Biomass proxy (OD600)",
                "CFP": "Reference fluorescence (CFP)",
                "YFP": "Reporter fluorescence (YFP)",
            },
            "show_replicates": True,
            "filename": "raw_kinetics",
        },
    }

    result = render_retron_experiment_plot(plot_spec, datasets={"ratio_yfp_od600/df": tidy})

    assert result.title == "Raw kinetics QC"
    assert len(result.figures) == 1
    axis_labels = [axis.get_ylabel() for axis in result.figures[0].fig.axes[:3]]
    assert axis_labels == [
        "Biomass proxy (OD600)",
        "Reference fluorescence (CFP)",
        "Reporter fluorescence (YFP)",
    ]
    assert result.figures[0].fig.axes[0].get_xlabel() == "Time from stress addition (h)"
    plt.close(result.figures[0].fig)


def test_render_retron_experiment_plot_cached_reuses_same_result_for_same_inputs() -> None:
    trace = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "tetO",
                "stress_condition": "H2O",
                "time_from_stress": 0.0,
                "metric": "C",
                "value": 0.0,
                "IPTG": "-IPTG",
                "relevant_sensor_pair": True,
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "stress_condition": "H2O",
                "time_from_stress": 1.0,
                "metric": "C",
                "value": -0.6,
                "IPTG": "+IPTG",
                "relevant_sensor_pair": True,
            },
        ]
    )
    plot_spec = {
        "id": "matched_control_kinetics",
        "plugin": "plot/retron_trace",
        "reads": {"trace": {"record": "semantic_metrics/trace"}},
        "with": {"metrics": ["C"], "filename": "matched_control_kinetics"},
    }

    first = render_retron_experiment_plot_cached(plot_spec, datasets={"semantic_metrics/trace": trace})
    second = render_retron_experiment_plot_cached(plot_spec, datasets={"semantic_metrics/trace": trace})

    assert second is first
    plt.close(first.figures[0].fig)


def test_render_retron_aggregate_plot_returns_specificity_matrix_bundle() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "S_AUC",
                "value": 0.40,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "sulAp",
                "sponge": "LexA",
                "metric": "S_AUC",
                "value": 0.55,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
        ]
    )

    result = render_retron_aggregate_plot(
        "specificity_matrix",
        summary_df=summary,
        sensor_target_map={"spyP": ("CpxR", "BaeR"), "sulAp": ("LexA",)},
        score_metric="S_AUC",
        architecture_x="irrelevant_motif_count",
        expected_mode="expected_sum",
        fingerprint_sponge=None,
    )

    assert result.title == "Specificity matrix"
    assert result.figure is not None
    assert "activity distributed across the sensors" in result.question
    assert result.figure.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert result.figure.axes[0].get_xlabel() == "Sponge design"
    assert result.figure.axes[0].get_ylabel() == ""
    assert not result.supporting_table.empty
    plt.close(result.figure)


def test_render_retron_aggregate_plot_cached_reuses_same_result_for_same_inputs() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "S_AUC",
                "value": 0.40,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            }
        ]
    )

    first = render_retron_aggregate_plot_cached(
        "specificity_matrix",
        summary_df=summary,
        sensor_target_map={"spyP": ("CpxR", "BaeR")},
        score_metric="S_AUC",
        architecture_x="irrelevant_motif_count",
        expected_mode="expected_sum",
        fingerprint_sponge=None,
    )
    second = render_retron_aggregate_plot_cached(
        "specificity_matrix",
        summary_df=summary,
        sensor_target_map={"spyP": ("CpxR", "BaeR")},
        score_metric="S_AUC",
        architecture_x="irrelevant_motif_count",
        expected_mode="expected_sum",
        fingerprint_sponge=None,
    )

    assert second is first
    assert first.figure is not None
    plt.close(first.figure)


def test_load_cached_parquet_frame_reuses_frame_when_file_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "frame.parquet"
    pd.DataFrame([{"value": 1}, {"value": 2}]).to_parquet(path, index=False)

    first = load_cached_parquet_frame(path)
    second = load_cached_parquet_frame(path)

    assert second is first


def test_filter_supporting_table_for_figure_uses_sensor_or_design_scope() -> None:
    summary_table = pd.DataFrame(
        [
            {"sensor": "spyP", "metric": "C_AUC", "value": 1.0},
            {"sensor": "sulAp", "metric": "C_AUC", "value": 2.0},
        ]
    )
    scoped_summary = filter_supporting_table_for_figure(
        summary_table,
        filename="matched_control_kinetics__sensor=spyp",
    )
    assert scoped_summary["sensor"].tolist() == ["spyP"]

    tidy_table = pd.DataFrame(
        [
            {"design_id_alias": "spyP/CpxR", "channel": "YFP", "value": 1.0},
            {"design_id_alias": "sulAp/LexA", "channel": "YFP", "value": 2.0},
        ]
    )
    scoped_tidy = filter_supporting_table_for_figure(
        tidy_table,
        filename="raw_kinetics__design_id_alias=spyP/CpxR",
    )
    assert scoped_tidy["design_id_alias"].tolist() == ["spyP/CpxR"]


def test_build_label_value_options_disambiguates_duplicate_labels() -> None:
    options = build_label_value_options(
        [
            {"Label": "Same label", "Value": "first"},
            {"Label": "Same label", "Value": "second"},
        ],
        label_key="Label",
        value_key="Value",
    )

    assert options == {
        "Same label [first]": "first",
        "Same label [second]": "second",
    }


def test_retron_table_kwargs_builds_compact_read_only_tables() -> None:
    kwargs = retron_table_kwargs(
        page_size=6,
        pagination=False,
        wrapped_columns=["Meaning"],
        max_height=320,
    )

    assert kwargs["selection"] is None
    assert kwargs["show_column_summaries"] is False
    assert kwargs["show_data_types"] is False
    assert kwargs["show_download"] is False
    assert kwargs["page_size"] == 6
    assert kwargs["pagination"] is False
    assert kwargs["wrapped_columns"] == ["Meaning"]
    assert kwargs["max_height"] == 320


def test_annotate_points_smart_spreads_labels_for_overlapping_points() -> None:
    figure, axis = plt.subplots()
    axis.scatter([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], s=30)

    annotations = annotate_points_smart(
        ax=axis,
        points=[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
        labels=["A", "B", "C"],
    )

    positions = [tuple(map(float, annotation.get_position())) for annotation in annotations]
    assert len(set(positions)) == len(positions)
    plt.close(figure)
