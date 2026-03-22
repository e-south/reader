from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import yaml

import reader.workbench.notebooks.retron_review as retron_review_mod
from reader.contracts import builtin_contract_catalog
from reader.domains.plate_reader.plots.common import annotate_points_smart
from reader.tests.support import base_reader_config, write_config
from reader.workbench.notebooks.retron_review import (
    RetronReviewBundle,
    RetronReviewSource,
    build_architecture_frame,
    build_expected_vs_observed_frame,
    build_fingerprint_frame,
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
    retron_aggregate_plot_rows,
    retron_experiment_plot_rows,
    retron_figure_coverage_rows,
    retron_plot_guide_rows,
    retron_plot_rendered_files,
    retron_source_selector_rows,
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


def test_retron_review_bundle_upgrades_legacy_absolute_metrics_for_aggregate_review(tmp_path: Path) -> None:
    legacy_summary = _write_csv(
        tmp_path / "exports" / "legacy_summary.csv",
        [
            {
                "plate_id": "plate_a",
                "sensor": "spyP",
                "sponge": "CpxR",
                "genotype_id": "spyP/CpxR",
                "stress_condition": "3% EtOH",
                "IPTG": pd.NA,
                "metric": "D_abs_AUC",
                "value": -0.12,
                "expected_decoy_sign": -1,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate_a",
                "sensor": "spyP",
                "sponge": "tetO",
                "genotype_id": pd.NA,
                "stress_condition": "3% EtOH",
                "IPTG": pd.NA,
                "metric": "G_sensor",
                "value": 2.0,
                "expected_decoy_sign": -1,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
        ],
    )
    legacy_trace = _write_csv(tmp_path / "exports" / "legacy_trace.csv", [{"metric": "D", "value": 0.1}])
    manifest_path = tmp_path / "review_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "relevant_stress_map": {"spyP": "3% EtOH"},
                "sensor_target_map": {"spyP": ["CpxR"]},
                "sources": [
                    {
                        "label": "legacy",
                        "experiment_id": "legacy_family",
                        "summary": str(legacy_summary.relative_to(tmp_path)),
                        "trace": str(legacy_trace.relative_to(tmp_path)),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    bundle = load_retron_review_bundle(manifest_path)
    o_abs_rows = bundle.summary_df[
        (bundle.summary_df["sensor"] == "spyP")
        & (bundle.summary_df["sponge"] == "CpxR")
        & (bundle.summary_df["metric"] == "O_abs_AUC")
    ]
    s_abs_rows = bundle.summary_df[
        (bundle.summary_df["sensor"] == "spyP")
        & (bundle.summary_df["sponge"] == "CpxR")
        & (bundle.summary_df["metric"] == "S_abs_AUC")
    ]

    assert len(o_abs_rows) == 1
    assert len(s_abs_rows) == 1
    assert float(o_abs_rows.iloc[0]["value"]) == pytest.approx(0.12)
    assert float(s_abs_rows.iloc[0]["value"]) == pytest.approx(0.06)
    assert bool(s_abs_rows.iloc[0]["scaling_available"]) is True
    assert float(s_abs_rows.iloc[0]["scale_reference_abs_g_sensor"]) == pytest.approx(2.0)
    assert float(s_abs_rows.iloc[0]["scale_min_abs_g_sensor"]) == pytest.approx(0.1)

    result = render_retron_aggregate_plot(
        "specificity_matrix",
        summary_df=bundle.summary_df,
        sensor_target_map={"spyP": ("CpxR",)},
        score_metric="O_abs_AUC",
        architecture_x="irrelevant_motif_count",
        expected_mode="expected_sum",
        fingerprint_sponge=None,
    )

    assert result.figure is not None
    assert not result.supporting_table.empty
    assert float(result.supporting_table.loc[0, "CpxR"]) == pytest.approx(0.12)
    plt.close(result.figure)


def test_load_retron_source_semantic_datasets_upgrades_legacy_trace_contract(tmp_path: Path) -> None:
    summary_path = _write_csv(
        tmp_path / "exports" / "summary.csv",
        [
            {
                "plate_id": "plate_a",
                "sensor": "spyP",
                "sponge": "CpxR",
                "genotype_id": "spyP/CpxR",
                "stress_condition": "3% EtOH",
                "IPTG": pd.NA,
                "metric": "D_abs_AUC",
                "value": -0.12,
                "expected_decoy_sign": -1,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate_a",
                "sensor": "spyP",
                "sponge": "CpxR",
                "genotype_id": "spyP/CpxR",
                "stress_condition": "3% EtOH",
                "IPTG": "+IPTG",
                "metric": "R_pre",
                "value": -3.7,
                "expected_decoy_sign": -1,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate_a",
                "sensor": "spyP",
                "sponge": "CpxR",
                "genotype_id": "spyP/CpxR",
                "stress_condition": "3% EtOH",
                "IPTG": "-IPTG",
                "metric": "R_pre",
                "value": -3.9,
                "expected_decoy_sign": -1,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate_a",
                "sensor": "spyP",
                "sponge": "tetO",
                "genotype_id": "spyP/tetO",
                "stress_condition": "3% EtOH",
                "IPTG": "+IPTG",
                "metric": "R_pre",
                "value": -3.8,
                "expected_decoy_sign": -1,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
            {
                "plate_id": "plate_a",
                "sensor": "spyP",
                "sponge": "tetO",
                "genotype_id": "spyP/tetO",
                "stress_condition": "3% EtOH",
                "IPTG": "-IPTG",
                "metric": "R_pre",
                "value": -4.0,
                "expected_decoy_sign": -1,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
        ],
    )
    trace_path = _write_csv(
        tmp_path / "exports" / "trace.csv",
        [
            {
                "plate_id": "plate_a",
                "acquisition_segment_id": "seg0",
                "sensor": "spyP",
                "sponge": "CpxR",
                "genotype_id": "spyP/CpxR",
                "replicate_id": "A1",
                "stress_condition": "3% EtOH",
                "IPTG": "-IPTG",
                "time": 0.0,
                "time_from_stress": -1.0,
                "metric": "R",
                "value": 0.2,
                "expected_decoy_sign": -1,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
                "matched_tetO_group": "plate_a::spyP::3% EtOH::-IPTG",
                "in_pre_window": True,
                "in_primary_post_stress": False,
                "in_endpoint_window": False,
                "configured_max_post_stress_hours": 4.0,
            },
            {
                "plate_id": "plate_a",
                "acquisition_segment_id": "seg1",
                "sensor": "spyP",
                "sponge": "CpxR",
                "genotype_id": "spyP/CpxR",
                "replicate_id": "A1",
                "stress_condition": "3% EtOH",
                "IPTG": "+IPTG",
                "time": 1.5,
                "time_from_stress": 0.5,
                "metric": "R",
                "value": 0.4,
                "expected_decoy_sign": -1,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
                "matched_tetO_group": "plate_a::spyP::3% EtOH::+IPTG",
                "in_pre_window": False,
                "in_primary_post_stress": True,
                "in_endpoint_window": True,
                "configured_max_post_stress_hours": 4.0,
            },
        ],
    )
    source = RetronReviewSource(
        label="legacy",
        experiment_id="legacy_source",
        experiment_root=None,
        config_path=None,
        summary_path=summary_path,
        trace_path=trace_path,
    )

    datasets = retron_review_mod.load_retron_source_semantic_datasets(
        source,
        record_ids=("semantic_metrics/summary", "semantic_metrics/trace"),
    )
    summary = datasets["semantic_metrics/summary"]
    trace = datasets["semantic_metrics/trace"]
    preload_rows = summary[
        (summary["sensor"] == "spyP") & (summary["sponge"] == "CpxR") & (summary["metric"] == "P_pre")
    ]

    assert len(preload_rows) == 1
    assert float(preload_rows.iloc[0]["value"]) == pytest.approx(0.0)
    assert "matched_control_key" in trace.columns
    assert "summary_window_start_h" in trace.columns
    assert "summary_window_end_h" in trace.columns
    assert "summary_window_duration_h" in trace.columns
    assert "pre_stress_read_count" in trace.columns
    assert "post_stress_read_count" in trace.columns
    assert "matched_group_sample_count" in trace.columns
    assert "stress_addition_gap_h" in trace.columns
    assert set(trace["matched_control_key"].astype(str)) == {"plate_a::spyP::3% EtOH"}
    assert float(trace["summary_window_start_h"].dropna().iloc[0]) == pytest.approx(0.5)
    assert float(trace["summary_window_end_h"].dropna().iloc[0]) == pytest.approx(0.5)
    assert float(trace["summary_window_duration_h"].dropna().iloc[0]) == pytest.approx(0.0)
    assert float(trace["pre_stress_read_count"].dropna().iloc[0]) == pytest.approx(1.0)
    assert float(trace["post_stress_read_count"].dropna().iloc[0]) == pytest.approx(1.0)
    assert float(trace["matched_group_sample_count"].dropna().iloc[0]) == pytest.approx(1.0)
    assert float(trace["stress_addition_gap_h"].dropna().iloc[0]) == pytest.approx(1.5)


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


def test_retron_review_bundle_rejects_non_list_sources(tmp_path: Path) -> None:
    manifest_path = tmp_path / "review_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "relevant_stress_map": {"spyP": "3% EtOH"},
                "sensor_target_map": {"spyP": ["CpxR", "BaeR"]},
                "sources": {"label": "mono"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest 'sources' must be a list"):
        load_retron_review_bundle(manifest_path)


def test_retron_review_bundle_rejects_non_mapping_source_entry(tmp_path: Path) -> None:
    manifest_path = tmp_path / "review_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "relevant_stress_map": {"spyP": "3% EtOH"},
                "sensor_target_map": {"spyP": ["CpxR", "BaeR"]},
                "sources": ["not-a-mapping"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"sources\[1\] must be a mapping"):
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


def test_retron_source_selector_rows_only_expand_duplicate_labels() -> None:
    dummy_summary = Path("outputs/exports/retron/semantic_summary.csv")
    dummy_trace = Path("outputs/exports/retron/semantic_trace.csv")
    bundle = RetronReviewBundle(
        manifest_path=Path("inputs/review_manifest.yaml"),
        sources=(
            RetronReviewSource(
                label="mono",
                experiment_id="exp_a",
                experiment_root=None,
                config_path=None,
                summary_path=dummy_summary,
                trace_path=dummy_trace,
            ),
            RetronReviewSource(
                label="mono",
                experiment_id="exp_b",
                experiment_root=None,
                config_path=None,
                summary_path=dummy_summary,
                trace_path=dummy_trace,
            ),
            RetronReviewSource(
                label="combo",
                experiment_id="exp_c",
                experiment_root=None,
                config_path=None,
                summary_path=dummy_summary,
                trace_path=dummy_trace,
            ),
        ),
        summary_df=pd.DataFrame(),
        trace_df=pd.DataFrame(),
        relevant_stress_map={},
        sensor_target_map={},
    )

    rows = retron_source_selector_rows(bundle)

    assert rows == [
        {"Selector label": "mono • exp_a", "Index": 0},
        {"Selector label": "mono • exp_b", "Index": 1},
        {"Selector label": "combo", "Index": 2},
    ]


def test_retron_figure_coverage_rows_call_out_cross_run_and_follow_on_views() -> None:
    coverage = retron_figure_coverage_rows()
    by_figure = {row["Figure"]: row for row in coverage}

    assert by_figure["Figure 13 — Target activity matrix"]["Coverage"] == "Exact aggregate notebook figure"
    assert by_figure["Figure 18 — Plate-position heatmaps"]["Coverage"] == "Not first-class compiled yet"


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
    selector_labels = {row["Plot id"]: row["Selector label"] for row in surface.plot_selector_rows}

    assert surface.experiment_title == "Source surface test"
    assert surface.protocol_id == "plate_reader/retron_sponge_screen"
    assert any(row["Plot id"] == "raw_kinetics" for row in surface.plot_catalog_rows)
    assert not any(row["Plot id"] == "support_kinetics" for row in surface.plot_selector_rows)
    assert not any(row["Plot id"] == "support_kinetics" for row in surface.plot_catalog_rows)
    assert not any(row["Plot id"] == "baseline_shifted_kinetics" for row in surface.plot_selector_rows)
    assert not any(row["Plot id"] == "baseline_shifted_kinetics" for row in surface.plot_catalog_rows)
    assert not any(row["Plot id"] == "matched_control_kinetics" for row in surface.plot_selector_rows)
    assert not any(row["Plot id"] == "matched_control_kinetics" for row in surface.plot_catalog_rows)
    assert not any(row["Plot id"] == "stress_modulation_scores" for row in surface.plot_selector_rows)
    assert not any(row["Plot id"] == "stress_modulation_scores" for row in surface.plot_catalog_rows)
    assert not any(row["Plot id"] == "pareto_ranking" for row in surface.plot_selector_rows)
    assert not any(row["Plot id"] == "pareto_ranking" for row in surface.plot_catalog_rows)
    assert surface.plot_selector_rows[0]["Plot id"] == "raw_kinetics"
    assert selector_labels["raw_kinetics"] == "QC raw channels"
    assert selector_labels["control_burden_panel"] == "Advanced QC / burden"
    assert selector_labels["induced_effect_kinetics"] == "Post-stress increment over time"
    assert selector_labels["absolute_effect_kinetics"] == "Total effect beyond matched tetO over time"
    assert selector_labels["control_anchored_decomposition"] == "Sponge vs matched tetO"
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


def test_retron_experiment_plot_rows_prioritize_matched_teto_summary() -> None:
    rows = retron_experiment_plot_rows(
        [
            {"id": "raw_kinetics", "with": {"title": "Raw kinetics"}},
            {"id": "absolute_effect_kinetics", "with": {"title": "Absolute effect"}},
            {"id": "control_anchored_decomposition", "with": {"title": "Sponge vs matched tetO"}},
        ]
    )

    assert [row["Plot id"] for row in rows] == [
        "raw_kinetics",
        "control_anchored_decomposition",
        "absolute_effect_kinetics",
    ]
    assert rows[0]["Selector label"] == "QC raw channels"
    assert rows[1]["Selector label"] == "Sponge vs matched tetO"


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


def test_retron_plot_guide_rows_share_registered_experiment_plot_copy() -> None:
    rows = retron_plot_guide_rows(["control_anchored_decomposition"])

    assert rows == [
        {
            "Stage": "2. Assay kinetics",
            "Plot": "Sponge vs matched tetO",
            "Plot id": "control_anchored_decomposition",
            "Math / transform": (
                "R(t)=log2(YFP/CFP)\n"
                "P_pre=delta_IPTG[R_pre-R_pre,tetO,matched]\n"
                "D_abs_AUC=AUC_window[D_abs(t)]\n"
                "D_AUC=AUC_window[D(t)]"
            ),
            "Source record": "semantic_metrics/trace",
            "How to read": (
                "Primary assay summary. Relevant-stress and H2O reporter-ratio traces show whether the sponge moves "
                "beyond matched tetO and whether that signal comes from preload, post-stress change, or both."
            ),
        }
    ]


def test_render_retron_experiment_plot_uses_explicit_iptg_state_wording_for_d_metric() -> None:
    rows: list[dict[str, object]] = []
    for time_value in (0.0, 1.0):
        for iptg, base in (("-IPTG", 0.2), ("+IPTG", -0.4)):
            for idx, delta in enumerate((0.00, 0.05), start=1):
                rows.append(
                    {
                        "sensor": "spyP",
                        "sponge": "CpxR",
                        "stress_condition": "3% EtOH",
                        "time_from_stress": time_value,
                        "metric": "C",
                        "value": base + time_value + delta,
                        "IPTG": iptg,
                        "replicate_id": f"r{idx}",
                        "is_relevant_stress": True,
                        "relevant_sensor_pair": True,
                        "expected_decoy_sign": -1,
                    }
                )
        rows.append(
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "stress_condition": "3% EtOH",
                "time_from_stress": time_value,
                "metric": "D",
                "value": -0.6,
                "IPTG": pd.NA,
                "replicate_id": pd.NA,
                "is_relevant_stress": True,
                "relevant_sensor_pair": True,
                "expected_decoy_sign": -1,
            }
        )
    trace = pd.DataFrame(rows)
    plot_spec = {
        "id": "induced_effect_kinetics",
        "plugin": "plot/retron_trace",
        "reads": {"trace": {"record": "semantic_metrics/trace"}},
        "with": {
            "metrics": ["D"],
            "title": "IPTG-state effect kinetics",
            "filename": "induced_effect_kinetics",
            "relevant_only": True,
        },
    }

    result = render_retron_experiment_plot(plot_spec, datasets={"semantic_metrics/trace": trace})

    assert result.title == "IPTG-state effect kinetics"
    assert "new movement appears after stress" in result.question
    assert result.meaning == (
        "Mechanistic view of the post-stress increment after preload removal, not the full IPTG-dependent effect."
    )
    assert set(result.supporting_table["metric"]) == {"D"}
    plt.close(result.figures[0].fig)


def test_render_retron_experiment_plot_rejects_unknown_plugin() -> None:
    plot_spec = {"id": "mystery_plot", "plugin": "plot/unknown", "reads": {}, "with": {}}

    with pytest.raises(ValueError, match="unsupported notebook plot plugin"):
        render_retron_experiment_plot(plot_spec, datasets={})


def test_render_retron_experiment_plot_rejects_missing_required_record_binding() -> None:
    plot_spec = {
        "id": "matched_control_kinetics",
        "plugin": "plot/retron_trace",
        "reads": {},
        "with": {"metrics": ["C"]},
    }

    with pytest.raises(ValueError, match="missing a record binding for 'trace'"):
        render_retron_experiment_plot(plot_spec, datasets={})


def test_render_retron_experiment_plot_rejects_non_mapping_with_field() -> None:
    plot_spec = {
        "id": "matched_control_kinetics",
        "plugin": "plot/retron_trace",
        "reads": {"trace": {"record": "semantic_metrics/trace"}},
        "with": ["not", "a", "mapping"],
    }

    with pytest.raises(ValueError, match=r"field 'with' must be a mapping"):
        render_retron_experiment_plot(plot_spec, datasets={})


def test_render_retron_experiment_plot_rejects_non_mapping_read_binding() -> None:
    plot_spec = {
        "id": "matched_control_kinetics",
        "plugin": "plot/retron_trace",
        "reads": {"trace": "semantic_metrics/trace"},
        "with": {"metrics": ["C"]},
    }

    with pytest.raises(ValueError, match=r"field 'reads.trace' must be a mapping"):
        render_retron_experiment_plot(plot_spec, datasets={})


def test_summary_supporting_table_uses_absolute_metric_family_for_heatmap_views() -> None:
    summary = pd.DataFrame(
        [
            {"sensor": "spyP", "sponge": "CpxR", "metric": "S_abs_AUC", "value": 0.8},
            {"sensor": "spyP", "sponge": "CpxR", "metric": "S_AUC", "value": 0.4},
            {"sensor": "spyP", "sponge": "CpxR", "metric": "P_pre", "value": 0.1},
            {"sensor": "spyP", "sponge": "CpxR", "metric": "M_AUC", "value": 0.2},
        ]
    )

    table = retron_review_mod._summary_supporting_table(
        summary,
        view="heatmap",
        metric="M_AUC",
        burden_metric="D_growth_AUC",
    )

    assert set(table["metric"]) == {"S_abs_AUC", "S_AUC", "P_pre"}


def test_summary_supporting_table_uses_configured_burden_metric_for_pareto_views() -> None:
    summary = pd.DataFrame(
        [
            {"sensor": "spyP", "sponge": "CpxR", "metric": "S_abs_AUC", "value": 0.8},
            {"sensor": "spyP", "sponge": "CpxR", "metric": "custom_burden", "value": 0.2},
            {"sensor": "spyP", "sponge": "CpxR", "metric": "L_pre", "value": 0.1},
            {"sensor": "spyP", "sponge": "CpxR", "metric": "P_pre", "value": 0.05},
            {"sensor": "spyP", "sponge": "CpxR", "metric": "D_growth_AUC", "value": 0.3},
        ]
    )

    table = retron_review_mod._summary_supporting_table(
        summary,
        view="pareto",
        metric="S_abs_AUC",
        burden_metric="custom_burden",
    )

    assert set(table["metric"]) == {"S_abs_AUC", "custom_burden", "L_pre", "P_pre"}


def test_summary_plot_config_defaults_pareto_burden_to_construct_specific_metric() -> None:
    config = retron_review_mod._summary_plot_config(
        plot_spec={"id": "pareto_ranking", "with": {}},
        with_cfg={},
    )

    assert config.burden_metric == "D_growth_AUC"


def test_render_retron_source_plot_cached_loads_overflow_context_for_qc_views(monkeypatch: pytest.MonkeyPatch) -> None:
    source = RetronReviewSource(
        label="mono",
        experiment_id="exp_retron_source_surface",
        experiment_root=None,
        config_path=None,
        summary_path=Path("summary.csv"),
        trace_path=Path("trace.csv"),
    )
    surface = retron_review_mod.RetronReviewSourceSurface(
        experiment_title="Source surface test",
        protocol_id="plate_reader/retron_sponge_screen",
        plot_specs=(
            {
                "id": "raw_kinetics",
                "plugin": "plot/time_series",
                "reads": {"df": {"record": "ratio_yfp_od600/df"}},
                "with": {},
            },
        ),
        plot_selector_rows=(),
        plot_catalog_rows=(),
        record_paths=(
            ("ratio_yfp_od600/df", "/tmp/ratio.parquet"),
            ("overflow/df", "/tmp/overflow.parquet"),
        ),
    )
    loaded_paths: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(retron_review_mod, "load_retron_source_surface", lambda _: surface)
    monkeypatch.setattr(
        retron_review_mod,
        "load_cached_parquet_frame",
        lambda path: loaded_paths.append(str(path)) or pd.DataFrame({"value": [1.0]}),
    )
    monkeypatch.setattr(
        retron_review_mod,
        "render_retron_experiment_plot_cached",
        lambda plot_spec, *, datasets: captured.update({"plot_spec": plot_spec, "datasets": datasets}) or "ok",
    )

    result = retron_review_mod.render_retron_source_plot_cached(source, plot_id="raw_kinetics")

    assert result == "ok"
    assert captured["plot_spec"]["id"] == "raw_kinetics"
    assert set(captured["datasets"]) == {"ratio_yfp_od600/df", "overflow/df"}
    assert loaded_paths == ["/tmp/ratio.parquet", "/tmp/overflow.parquet"]


def test_render_retron_source_plot_cached_prefers_semantic_exports_for_retron_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = RetronReviewSource(
        label="mono",
        experiment_id="exp_retron_source_surface",
        experiment_root=None,
        config_path=None,
        summary_path=Path("summary.csv"),
        trace_path=Path("trace.csv"),
    )
    surface = retron_review_mod.RetronReviewSourceSurface(
        experiment_title="Source surface test",
        protocol_id="plate_reader/retron_sponge_screen",
        plot_specs=(
            {
                "id": "control_anchored_decomposition",
                "plugin": "plot/retron_summary",
                "reads": {
                    "summary": {"record": "semantic_metrics/summary"},
                    "trace": {"record": "semantic_metrics/trace"},
                },
                "with": {"view": "decomposition", "metric": "D_abs_AUC", "control_name": "tetO"},
            },
        ),
        plot_selector_rows=(),
        plot_catalog_rows=(),
        record_paths=(
            ("semantic_metrics/summary", "/tmp/stale_summary.parquet"),
            ("semantic_metrics/trace", "/tmp/stale_trace.parquet"),
        ),
    )
    loaded_paths: list[str] = []
    captured: dict[str, object] = {}
    semantic_datasets = {
        "semantic_metrics/summary": pd.DataFrame({"metric": ["D_abs_AUC"], "value": [0.1]}),
        "semantic_metrics/trace": pd.DataFrame({"metric": ["R"], "value": [0.2]}),
    }

    monkeypatch.setattr(retron_review_mod, "load_retron_source_surface", lambda _: surface)
    monkeypatch.setattr(
        retron_review_mod,
        "load_retron_source_semantic_datasets",
        lambda source, *, record_ids=None: {
            record_id: semantic_datasets[record_id] for record_id in (record_ids or tuple(semantic_datasets))
        },
    )
    monkeypatch.setattr(
        retron_review_mod,
        "load_cached_parquet_frame",
        lambda path: loaded_paths.append(str(path)) or pd.DataFrame({"value": [1.0]}),
    )
    monkeypatch.setattr(
        retron_review_mod,
        "render_retron_experiment_plot_cached",
        lambda plot_spec, *, datasets: captured.update({"plot_spec": plot_spec, "datasets": datasets}) or "ok",
    )

    result = retron_review_mod.render_retron_source_plot_cached(
        source,
        plot_id="control_anchored_decomposition",
    )

    assert result == "ok"
    assert captured["plot_spec"]["id"] == "control_anchored_decomposition"
    assert captured["datasets"] == semantic_datasets
    assert loaded_paths == []


def test_render_retron_source_plot_cached_rejects_missing_dataframe_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = RetronReviewSource(
        label="mono",
        experiment_id="exp_retron_source_surface",
        experiment_root=None,
        config_path=None,
        summary_path=Path("summary.csv"),
        trace_path=Path("trace.csv"),
    )
    surface = retron_review_mod.RetronReviewSourceSurface(
        experiment_title="Source surface test",
        protocol_id="plate_reader/retron_sponge_screen",
        plot_specs=(
            {
                "id": "raw_kinetics",
                "plugin": "plot/time_series",
                "reads": {"df": {"record": "ratio_yfp_od600/df"}},
                "with": {},
            },
        ),
        plot_selector_rows=(),
        plot_catalog_rows=(),
        record_paths=(),
    )

    monkeypatch.setattr(retron_review_mod, "load_retron_source_surface", lambda _: surface)

    with pytest.raises(ValueError, match="Missing dataframe record `ratio_yfp_od600/df`"):
        retron_review_mod.render_retron_source_plot_cached(source, plot_id="raw_kinetics")


def test_render_retron_experiment_plot_surfaces_control_anchored_decomposition_frame() -> None:
    rows: list[dict[str, object]] = []
    for sponge, control_flag, minus_values, plus_values in (
        ("LexA", False, (0.85, 0.88), (1.28, 1.31)),
        ("tetO", True, (0.72, 0.75), (0.90, 0.92)),
    ):
        for iptg, values in (("-IPTG", minus_values), ("+IPTG", plus_values)):
            for idx, value in enumerate(values, start=1):
                for time_value in (0.0, 4.0):
                    rows.append(
                        {
                            "plate_id": "plate-1",
                            "sensor": "sulAp",
                            "sponge": sponge,
                            "stress_condition": "100 nM ciprofloxacin",
                            "time_from_stress": time_value,
                            "metric": "R",
                            "value": value + (0.10 if time_value > 0 else 0.0),
                            "IPTG": iptg,
                            "replicate_id": f"r{idx}",
                            "is_relevant_stress": True,
                            "relevant_sensor_pair": not control_flag,
                            "in_primary_post_stress": True,
                            "configured_max_post_stress_hours": 4.0,
                            "matched_control_key": "plate-1::sulAp::100 nM ciprofloxacin",
                            "summary_window_start_h": 0.0,
                            "summary_window_end_h": 4.0,
                            "summary_window_duration_h": 4.0,
                            "pre_stress_read_count": 1,
                            "post_stress_read_count": 2,
                            "matched_group_sample_count": 2,
                            "stress_addition_gap_h": 0.5,
                        }
                    )
    trace = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "P_pre",
                "value": 0.12,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_abs_AUC",
                "value": 0.42,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_AUC",
                "value": 0.30,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_growth_AUC",
                "value": -0.08,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "sulAp",
                "sponge": "tetO",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "G_sensor",
                "value": 0.60,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
        ]
    )
    plot_spec = {
        "id": "control_anchored_decomposition",
        "plugin": "plot/retron_summary",
        "reads": {
            "summary": {"record": "semantic_metrics/summary"},
            "trace": {"record": "semantic_metrics/trace"},
        },
        "with": {
            "view": "decomposition",
            "metric": "D_abs_AUC",
            "title": "Sponge vs matched tetO",
            "filename": "control_anchored_decomposition",
            "control_name": "tetO",
            "relevant_only": True,
        },
    }

    result = render_retron_experiment_plot(
        plot_spec,
        datasets={"semantic_metrics/summary": summary, "semantic_metrics/trace": trace},
    )

    assert result.title == "Sponge vs matched tetO"
    assert "matched tetO" in result.question
    assert result.math.startswith("R(t)=log2(YFP/CFP)")
    assert result.meaning.startswith("Primary assay summary.")
    assert {
        "panel_role",
        "primary_stress",
        "matched_control_key",
        "matched_group_sample_count",
        "summary_window_duration_h",
        "stress_addition_gap_h",
        "summary_metric",
        "summary_label",
        "estimate",
        "lower",
        "upper",
        "units",
    } <= set(result.supporting_table.columns)
    assert "sample_minus_auc" not in result.supporting_table.columns
    assert set(result.supporting_table["summary_metric"]) == {
        "P_pre",
        "D_abs_AUC",
        "D_AUC",
        "D_growth_AUC",
    }
    plt.close(result.figures[0].fig)


def test_render_retron_experiment_plot_rejects_stale_decision_card_trace_contract() -> None:
    trace = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "time_from_stress": 0.0,
                "metric": "R",
                "value": 0.9,
                "IPTG": "-IPTG",
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {"sensor": "sulAp", "sponge": "LexA", "metric": "P_pre", "value": 0.1},
            {"sensor": "sulAp", "sponge": "LexA", "metric": "D_abs_AUC", "value": 0.3},
            {"sensor": "sulAp", "sponge": "LexA", "metric": "D_AUC", "value": 0.2},
            {"sensor": "sulAp", "sponge": "LexA", "metric": "D_growth_AUC", "value": -0.05},
            {"sensor": "sulAp", "sponge": "tetO", "metric": "G_sensor", "value": 0.4},
        ]
    )
    plot_spec = {
        "id": "control_anchored_decomposition",
        "plugin": "plot/retron_summary",
        "reads": {
            "summary": {"record": "semantic_metrics/summary"},
            "trace": {"record": "semantic_metrics/trace"},
        },
        "with": {
            "view": "decomposition",
            "metric": "D_abs_AUC",
            "control_name": "tetO",
            "relevant_only": True,
        },
    }

    with pytest.raises(ValueError, match="requires refreshed semantic trace records"):
        render_retron_experiment_plot(
            plot_spec,
            datasets={"semantic_metrics/summary": summary, "semantic_metrics/trace": trace},
        )


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

    assert result.title == "QC raw channels"
    assert len(result.figures) == 1
    axis_labels = [axis.get_ylabel() for axis in result.figures[0].fig.axes if axis.get_visible()]
    assert axis_labels == ["OD600", "YFP", "CFP"]
    assert result.figures[0].fig.axes[0].get_xlabel() == "Time from stress addition (h)"
    plt.close(result.figures[0].fig)


def test_render_retron_experiment_plot_prefers_overflow_surface_for_raw_cfp_rows() -> None:
    overflow_df = pd.DataFrame(
        [
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 0.0,
                "sheet_index": 0,
                "overflow": False,
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
                "overflow": False,
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
                "overflow": False,
                "channel": "CFP",
                "value": 8000.0,
            },
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 1.0,
                "sheet_index": 0,
                "overflow": True,
                "channel": "CFP",
                "value": 9000.0,
            },
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 0.0,
                "sheet_index": 0,
                "overflow": False,
                "channel": "YFP",
                "value": 1200.0,
            },
            {
                "design_id": "spyP/CpxR",
                "design_id_alias": "spyP/CpxR",
                "treatment": "0 uM IPTG",
                "treatment_alias": "-IPTG/-stress",
                "time": 1.0,
                "sheet_index": 0,
                "overflow": False,
                "channel": "YFP",
                "value": 1800.0,
            },
        ]
    )
    ratio_df = pd.concat(
        [
            overflow_df.assign(value=lambda frame: frame["value"].where(frame["channel"] != "CFP", 96117.196)),
            pd.DataFrame(
                [
                    {
                        "design_id": "spyP/CpxR",
                        "design_id_alias": "spyP/CpxR",
                        "treatment": "0 uM IPTG",
                        "treatment_alias": "-IPTG/-stress",
                        "time": 0.0,
                        "sheet_index": 0,
                        "overflow": False,
                        "channel": "YFP/OD600",
                        "value": 12000.0,
                    },
                    {
                        "design_id": "spyP/CpxR",
                        "design_id_alias": "spyP/CpxR",
                        "treatment": "0 uM IPTG",
                        "treatment_alias": "-IPTG/-stress",
                        "time": 1.0,
                        "sheet_index": 0,
                        "overflow": False,
                        "channel": "YFP/OD600",
                        "value": 9000.0,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    plot_spec = {
        "id": "raw_kinetics",
        "plugin": "plot/time_series",
        "reads": {"df": {"record": "ratio_yfp_od600/df"}},
        "with": {
            "partition": {"by": "design_id"},
            "hue": "treatment",
            "xlabel": "Time from stress addition (h)",
            "filename": "raw_kinetics",
        },
    }

    result = render_retron_experiment_plot(
        plot_spec,
        datasets={
            "ratio_yfp_od600/df": ratio_df,
            "overflow/df": overflow_df,
        },
    )

    cfp_rows = result.supporting_table[result.supporting_table["channel"] == "CFP"].sort_values("time")
    assert cfp_rows["value"].tolist() == [8000.0, 9000.0]
    assert cfp_rows["overflow"].tolist() == [False, True]
    assert result.source_record == "overflow/df for raw OD600/YFP/CFP; ratio_yfp_od600/df for support ratios"
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

    assert result.title == "Target activity matrix"
    assert result.figure is not None
    assert "strongest total effect in the expected direction" in result.question
    assert "score(sensor,sponge)=mean_source[O_abs_AUC or S_abs_AUC]" in result.math
    assert "without implying exhaustive off-target specificity coverage" in result.meaning
    assert result.figure.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert result.figure.axes[0].get_xlabel() == "Sponge design"
    assert result.figure.axes[0].get_title() == "Relevant-stress target activity matrix"
    assert result.figure.axes[0].get_ylabel() == ""
    assert result.figure.axes[1].get_ylabel() == "Scaled expected-direction increment"
    assert not result.supporting_table.empty
    assert result.supporting_table_title == "Relevant-stress on-target matrix behind the heatmap"
    plt.close(result.figure)


def test_render_retron_aggregate_plot_rejects_unavailable_primary_metric() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "O_AUC",
                "value": 0.40,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            }
        ]
    )

    with pytest.raises(ValueError, match="aggregate score metric 'O_abs_AUC' is unavailable"):
        render_retron_aggregate_plot(
            "specificity_matrix",
            summary_df=summary,
            sensor_target_map={"spyP": ("CpxR", "BaeR")},
            score_metric="O_abs_AUC",
            architecture_x="irrelevant_motif_count",
            expected_mode="expected_sum",
            fingerprint_sponge=None,
        )


def test_render_retron_aggregate_plot_returns_pareto_bundle() -> None:
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
                "sensor": "soxSp",
                "sponge": "SoxR-SoxS",
                "metric": "S_AUC",
                "value": 0.55,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "L_pre",
                "value": 0.02,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxR-SoxS",
                "metric": "L_pre",
                "value": -0.03,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "D_growth_AUC",
                "value": -0.01,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxR-SoxS",
                "metric": "D_growth_AUC",
                "value": 0.04,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
        ]
    )

    result = render_retron_aggregate_plot(
        "pareto_ranking",
        summary_df=summary,
        sensor_target_map={"spyP": ("CpxR",), "soxSp": ("SoxR", "SoxS")},
        score_metric="S_AUC",
        architecture_x="irrelevant_motif_count",
        expected_mode="expected_sum",
        fingerprint_sponge=None,
    )

    assert result.title == "Pareto ranking"
    assert result.figure is not None
    assert not result.supporting_table.empty
    assert set(result.supporting_table["sponge"]) == {"CpxR", "SoxR-SoxS"}
    assert "x=mean[O_abs_AUC or S_abs_AUC]" in result.math
    assert "y=mean[-D_growth_AUC]" in result.math
    assert "total effect, burden, and leakiness in view" in result.meaning
    assert result.supporting_table_title == "Aggregate on-target, burden, and leakiness table for candidate ranking"
    x_tick_sizes = {label.get_fontsize() for label in result.figure.axes[0].get_xticklabels() if label.get_text()}
    y_tick_sizes = {label.get_fontsize() for label in result.figure.axes[0].get_yticklabels() if label.get_text()}
    legend = result.figure.axes[0].get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"mono", "bi"}
    assert x_tick_sizes == {7.0}
    assert y_tick_sizes == {7.0}
    plt.close(result.figure)


def test_retron_aggregate_plot_rows_reject_unknown_plot_id() -> None:
    with pytest.raises(ValueError, match="unknown aggregate plot id"):
        retron_aggregate_plot_rows(["unknown_plot"])


def test_build_fingerprint_frame_preserves_source_replicates_and_matched_teto_rows() -> None:
    summary = pd.DataFrame(
        [
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "spyP",
                "sponge": "CpxR-LexA",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.42,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "sulAp",
                "sponge": "CpxR-LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "S_AUC",
                "value": 0.36,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "spyP",
                "sponge": "tetO",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.01,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "sulAp",
                "sponge": "tetO",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "S_AUC",
                "value": -0.02,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
            {
                "source_experiment_id": "exp_b",
                "source_label": "quad",
                "sensor": "spyP",
                "sponge": "CpxR-LexA",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.47,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "source_experiment_id": "exp_b",
                "source_label": "quad",
                "sensor": "sulAp",
                "sponge": "CpxR-LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "S_AUC",
                "value": 0.29,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "source_experiment_id": "exp_b",
                "source_label": "quad",
                "sensor": "spyP",
                "sponge": "tetO",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.03,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
            {
                "source_experiment_id": "exp_b",
                "source_label": "quad",
                "sensor": "sulAp",
                "sponge": "tetO",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "S_AUC",
                "value": -0.01,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
        ]
    )

    frame = build_fingerprint_frame(summary, score_metric="S_AUC", fingerprint_sponge="CpxR-LexA")

    assert set(frame["comparison_group"]) == {"Selected sponge", "tetO reference"}
    assert set(frame["source_experiment_id"]) == {"exp_a", "exp_b"}
    assert set(frame["sensor"]) == {"spyP", "sulAp"}
    assert frame["selected_sponge"].tolist() == ["CpxR-LexA"] * len(frame)
    assert len(frame) == 8


def test_build_fingerprint_frame_rejects_unknown_requested_sponge() -> None:
    summary = pd.DataFrame(
        [
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "spyP",
                "sponge": "CpxR-LexA",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.42,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            }
        ]
    )

    with pytest.raises(ValueError, match="requested fingerprint sponge"):
        build_fingerprint_frame(summary, score_metric="S_AUC", fingerprint_sponge="BaeR-LexA")


def test_render_retron_aggregate_plot_returns_grouped_fingerprint_bundle() -> None:
    summary = pd.DataFrame(
        [
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "spyP",
                "sponge": "CpxR-LexA",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.42,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "sulAp",
                "sponge": "CpxR-LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "S_AUC",
                "value": 0.36,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "spyP",
                "sponge": "tetO",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.01,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "sulAp",
                "sponge": "tetO",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "S_AUC",
                "value": -0.02,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
            {
                "source_experiment_id": "exp_b",
                "source_label": "quad",
                "sensor": "spyP",
                "sponge": "CpxR-LexA",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.47,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "source_experiment_id": "exp_b",
                "source_label": "quad",
                "sensor": "sulAp",
                "sponge": "CpxR-LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "S_AUC",
                "value": 0.29,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "source_experiment_id": "exp_b",
                "source_label": "quad",
                "sensor": "spyP",
                "sponge": "tetO",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.03,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
            {
                "source_experiment_id": "exp_b",
                "source_label": "quad",
                "sensor": "sulAp",
                "sponge": "tetO",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "S_AUC",
                "value": -0.01,
                "relevant_sensor_pair": False,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
        ]
    )

    result = render_retron_aggregate_plot(
        "sponge_fingerprint",
        summary_df=summary,
        sensor_target_map={"spyP": ("CpxR", "BaeR"), "sulAp": ("LexA",)},
        score_metric="S_AUC",
        architecture_x="irrelevant_motif_count",
        expected_mode="expected_sum",
        fingerprint_sponge="CpxR-LexA",
    )

    assert result.title == "Sponge fingerprint"
    assert result.figure is not None
    assert set(result.supporting_table["comparison_group"]) == {"Selected sponge", "tetO reference"}
    axis = result.figure.axes[0]
    assert axis.get_title() == "CpxR-LexA"
    assert [label.get_text() for label in axis.get_xticklabels()] == ["spyP", "sulAp"]
    assert len(axis.patches) == 4
    assert len(axis.collections) >= 4
    legend = axis.get_legend()
    assert legend is not None
    assert any("Bars show source means" in text.get_text() for text in result.figure.texts)
    assert {text.get_text() for text in legend.get_texts()} == {"Selected sponge", "tetO reference"}
    plt.close(result.figure)


def test_render_retron_aggregate_plot_rejects_unknown_fingerprint_sponge() -> None:
    summary = pd.DataFrame(
        [
            {
                "source_experiment_id": "exp_a",
                "source_label": "tri",
                "sensor": "spyP",
                "sponge": "CpxR-LexA",
                "stress_condition": "3% EtOH",
                "metric": "S_AUC",
                "value": 0.42,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            }
        ]
    )

    with pytest.raises(ValueError, match="requested fingerprint sponge"):
        render_retron_aggregate_plot(
            "sponge_fingerprint",
            summary_df=summary,
            sensor_target_map={"spyP": ("CpxR", "BaeR")},
            score_metric="S_AUC",
            architecture_x="irrelevant_motif_count",
            expected_mode="expected_sum",
            fingerprint_sponge="BaeR-LexA",
        )


def test_render_retron_aggregate_architecture_plot_uses_shared_subplot_limits() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "S_AUC",
                "value": 0.20,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR-LexA",
                "metric": "S_AUC",
                "value": -0.25,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxR",
                "metric": "S_AUC",
                "value": 0.10,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "BaeR-SoxR",
                "metric": "S_AUC",
                "value": 0.35,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
        ]
    )

    result = render_retron_aggregate_plot(
        "architecture_plot",
        summary_df=summary,
        sensor_target_map={"spyP": ("CpxR", "BaeR"), "soxSp": ("SoxR", "SoxS")},
        score_metric="S_AUC",
        architecture_x="irrelevant_motif_count",
        expected_mode="expected_sum",
        fingerprint_sponge=None,
    )

    assert result.figure is not None
    axes = [axis for axis in result.figure.axes if axis.get_visible()]
    assert axes[0].get_xlim() == pytest.approx(axes[1].get_xlim())
    assert axes[0].get_ylim() == pytest.approx(axes[1].get_ylim())
    plt.close(result.figure)


def test_render_retron_aggregate_expected_vs_observed_uses_shared_square_limits() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "S_AUC",
                "value": 0.20,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR-LexA",
                "metric": "S_AUC",
                "value": -0.10,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "sulAp",
                "sponge": "LexA",
                "metric": "S_AUC",
                "value": 0.12,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "sulAp",
                "sponge": "CpxR-LexA",
                "metric": "S_AUC",
                "value": 0.05,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxR",
                "metric": "S_AUC",
                "value": 0.08,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "CpxR-SoxR",
                "metric": "S_AUC",
                "value": 0.18,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
        ]
    )

    result = render_retron_aggregate_plot(
        "expected_vs_observed",
        summary_df=summary,
        sensor_target_map={"spyP": ("CpxR", "BaeR"), "sulAp": ("LexA",), "soxSp": ("SoxR", "SoxS")},
        score_metric="S_AUC",
        architecture_x="irrelevant_motif_count",
        expected_mode="expected_best_single",
        fingerprint_sponge=None,
    )

    assert result.figure is not None
    axes = [axis for axis in result.figure.axes if axis.get_visible()]
    assert axes[0].get_xlim() == pytest.approx(axes[1].get_xlim())
    assert axes[0].get_xlim() == pytest.approx(axes[0].get_ylim())
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
