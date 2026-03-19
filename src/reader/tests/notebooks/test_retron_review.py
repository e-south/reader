from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from reader.tests.support import base_reader_config, write_config
from reader.workbench.notebooks.retron_review import (
    build_architecture_frame,
    build_expected_vs_observed_frame,
    build_specificity_matrix,
    load_notebook_workbench_context,
    load_retron_review_bundle,
    retron_figure_coverage_rows,
    retron_plot_rendered_files,
)


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
    assert float(specificity.loc["BaeR-LexA-SoxR-SoxS", "soxSp"]) == pytest.approx(0.48)
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


def test_retron_plot_rendered_files_accepts_legacy_raw_kinetics_prefix(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "ts_spyP_tetO.pdf").write_text("legacy", encoding="utf-8")

    matches = retron_plot_rendered_files(plots_dir, plot_id="raw_kinetics", plugin="plot/time_series")

    assert matches == ["ts_spyP_tetO.pdf"]
