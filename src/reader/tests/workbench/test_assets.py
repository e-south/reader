from __future__ import annotations

import reader.workbench.assets as assets
from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.workbench.assets import static_asset_catalog
from reader.workbench.assets.plugin_manifest import (
    builtin_plugin_asset_catalog,
    builtin_plugin_descriptors,
)
from reader.workbench.templates import select_default_notebook_template


def test_select_default_notebook_template_uses_protocol_policy() -> None:
    protocols = builtin_protocol_catalog()
    assert (
        select_default_notebook_template(
            protocol=protocols.bind(ProtocolBinding(id="plate_reader/dual_reporter_screen"))
        ).template
        == "notebook/eda"
    )
    assert (
        select_default_notebook_template(protocol=protocols.bind(ProtocolBinding(id="cytometry/flow_panel"))).template
        == "notebook/cytometry"
    )
    assert (
        select_default_notebook_template(protocol=protocols.bind(ProtocolBinding(id="workbench/generic"))).template
        == "notebook/eda"
    )


def test_static_asset_catalog_only_exposes_templates() -> None:
    catalog = static_asset_catalog()
    items = catalog.all()
    assert items
    assert {item.kind for item in items} == {"template"}
    assert catalog.resolve("notebook/dual_reporter_triptych", kind="template").kind == "template"


def test_sample_metadata_descriptor_is_not_plate_reader_specific() -> None:
    descriptor = builtin_plugin_asset_catalog().resolve("transform/sample_metadata", kind="plugin")

    assert descriptor.domain == "generic"
    assert descriptor.summary == "Attach sample-keyed metadata tables to tidy measurement rows."


def test_builtin_plugin_manifest_preserves_the_complete_plugin_id_set() -> None:
    assert {descriptor.plugin_id for descriptor in builtin_plugin_descriptors()} == {
        "export/csv",
        "export/xlsx",
        "ingest/flow_cytometer",
        "ingest/synergy_h1",
        "plot/distributions",
        "plot/logic_symmetry",
        "plot/response_window_summary",
        "plot/sfxi_vec8_collection",
        "plot/sfxi_vec8_heatmap",
        "plot/snapshot_barplot",
        "plot/snapshot_heatmap",
        "plot/time_series",
        "plot/ts_and_snap",
        "transform/alias",
        "transform/assay_labels",
        "transform/blank_correction",
        "transform/crosstalk_pairs",
        "transform/fold_change",
        "transform/logic_symmetry",
        "transform/outlier_filter",
        "transform/overflow_handling",
        "transform/ratio",
        "transform/response_window",
        "transform/sample_map",
        "transform/sample_metadata",
        "transform/sfxi",
        "transform/sfxi_vec8_collection",
        "validator/to_tidy_plus_map",
    }


def test_asset_package_does_not_publish_a_duplicate_plugin_template_catalog() -> None:
    assert "build_workbench_asset_catalog" not in assets.__all__
    assert not hasattr(assets, "build_workbench_asset_catalog")
