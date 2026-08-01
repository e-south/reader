from __future__ import annotations

import reader_workbench.workbench.assets as assets
from reader_workbench.plugins.catalog import builtin_plugin_catalog, builtin_plugin_descriptors


def test_workbench_asset_surface_is_plugin_only() -> None:
    assert "static_asset_catalog" not in assets.__all__
    assert "AssetKind" not in assets.__all__
    assert not hasattr(assets, "static_asset_catalog")
    assert not hasattr(assets, "AssetKind")


def test_sample_metadata_descriptor_is_not_plate_reader_specific() -> None:
    descriptor = builtin_plugin_catalog().resolve("transform/sample_metadata")

    assert not hasattr(descriptor, "kind")
    assert not hasattr(descriptor, "template")
    assert not hasattr(descriptor, "body")
    assert descriptor.domain == "generic"
    assert descriptor.summary == "Attach sample-keyed metadata tables to tidy measurement rows."


def test_response_window_diagnostic_descriptor_is_source_design_scoped() -> None:
    descriptor = builtin_plugin_catalog().resolve("plot/response_window_diagnostic")

    assert descriptor.domain == "plate_reader"
    assert descriptor.family == "event_relative_diagnostic"
    assert (
        descriptor.summary == "Render trajectories and reduced components for one explicitly identified source design."
    )


def test_dual_reporter_triptych_descriptor_is_plate_reader_plot() -> None:
    descriptor = builtin_plugin_catalog().resolve("plot/dual_reporter_triptych")

    assert descriptor.domain == "plate_reader"
    assert descriptor.family == "composite_plot"
    assert descriptor.summary == "Render per-design growth, reporter-ratio, and endpoint panels."


def test_single_reporter_diagnostic_descriptor_is_study_neutral_plate_reader_plot() -> None:
    descriptor = builtin_plugin_catalog().resolve("plot/single_reporter_diagnostic")

    assert descriptor.domain == "plate_reader"
    assert descriptor.family == "composite_plot"
    assert descriptor.summary == (
        "Render single-reporter kinetics, an explicit reduction, and normalizer QC in one row."
    )


def test_builtin_plugin_manifest_preserves_the_complete_plugin_id_set() -> None:
    assert {descriptor.plugin_id for descriptor in builtin_plugin_descriptors()} == {
        "export/csv",
        "export/xlsx",
        "ingest/flow_cytometer",
        "ingest/synergy_h1",
        "plot/distributions",
        "plot/cytometry_diagnostic",
        "plot/dual_reporter_triptych",
        "plot/logic_symmetry",
        "plot/response_window_diagnostic",
        "plot/response_window_summary",
        "plot/four_state_vector_diagnostic",
        "plot/four_state_vector_collection",
        "plot/four_state_vector_heatmap",
        "plot/single_reporter_diagnostic",
        "plot/snapshot_barplot",
        "plot/snapshot_heatmap",
        "plot/time_series",
        "plot/ts_and_snap",
        "transform/alias",
        "transform/assay_labels",
        "transform/blank_correction",
        "transform/crosstalk_pairs",
        "transform/cytometry_gating",
        "transform/fold_change",
        "transform/logic_symmetry",
        "transform/outlier_filter",
        "transform/overflow_handling",
        "transform/ratio",
        "transform/response_window",
        "transform/sample_map",
        "transform/sample_metadata",
        "transform/four_state_vector",
        "transform/four_state_vector_collection",
        "validator/to_tidy_plus_map",
    }


def test_asset_package_does_not_publish_a_duplicate_plugin_template_catalog() -> None:
    assert "build_workbench_asset_catalog" not in assets.__all__
    assert not hasattr(assets, "build_workbench_asset_catalog")


def test_asset_package_does_not_publish_template_applicability_as_capabilities() -> None:
    assert "AssetCapabilities" not in assets.__all__
    assert "AssetRequirement" not in assets.__all__
    assert not hasattr(assets, "AssetCapabilities")
    assert not hasattr(assets, "AssetRequirement")
