from __future__ import annotations

from reader.plugins.export.csv import ExportCsv
from reader.plugins.export.xlsx import ExportXlsx
from reader.plugins.ingest.flow_cytometer import FlowCytometerIngest
from reader.plugins.ingest.synergy_h1 import SynergyH1
from reader.plugins.plot.distributions import DistributionsPlot
from reader.plugins.plot.logic_symmetry import LogicSymmetryPlot
from reader.plugins.plot.snapshot_barplot import SnapshotBarplot
from reader.plugins.plot.snapshot_heatmap import SnapshotHeatmapPlot
from reader.plugins.plot.time_series import TimeSeriesPlot
from reader.plugins.plot.ts_and_snap import TSAndSnapPlot
from reader.plugins.transform.alias import AliasTransform
from reader.plugins.transform.assay_labels import AnnotationLabelsTransform
from reader.plugins.transform.blank import BlankCorrection
from reader.plugins.transform.crosstalk_pairs import CrosstalkPairs
from reader.plugins.transform.fold_change import FoldChange
from reader.plugins.transform.outlier_filter import OutlierFilter
from reader.plugins.transform.overflow import OverflowHandling
from reader.plugins.transform.ratio import RatioTransform
from reader.plugins.transform.retron_sponge_metrics import RetronSpongeMetrics
from reader.plugins.transform.sample_map import SampleMapMerge
from reader.plugins.transform.sample_metadata import SampleMetadataMerge
from reader.plugins.transform.sfxi import SFXITransform
from reader.plugins.validator.to_tidy_plus_map import PromoteToTidyPlusMap
from reader.workbench.ontology import PluginSemantics

from .types import AssetCatalog, AssetDescriptor, build_plugin_asset

_BUILTIN_PLUGIN_CATALOG = AssetCatalog(
    [
        build_plugin_asset(
            plugin_id="ingest/flow_cytometer",
            semantics=PluginSemantics(
                domain="cytometry",
                family="fcs_ingest",
                summary="Parse FCS cytometry files into tidy event tables and channel metadata.",
                tags=("fcs", "events", "channels"),
            ),
            plugin_cls=FlowCytometerIngest,
        ),
        build_plugin_asset(
            plugin_id="transform/alias",
            semantics=PluginSemantics(
                domain="generic",
                family="label_enrichment",
                summary="Add alias columns for configured categorical metadata.",
                tags=("aliases", "annotation"),
            ),
            plugin_cls=AliasTransform,
        ),
        build_plugin_asset(
            plugin_id="transform/assay_labels",
            semantics=PluginSemantics(
                domain="generic",
                family="label_enrichment",
                summary="Materialize configured annotations.labels into dataframe columns.",
                tags=("annotations", "labels", "annotation"),
            ),
            plugin_cls=AnnotationLabelsTransform,
        ),
        build_plugin_asset(
            plugin_id="transform/outlier_filter",
            semantics=PluginSemantics(
                domain="generic",
                family="quality_filter",
                summary="Drop outlier tidy rows using per-channel and per-time z-scores.",
                tags=("qc", "filtering"),
            ),
            plugin_cls=OutlierFilter,
        ),
        build_plugin_asset(
            plugin_id="export/csv",
            semantics=PluginSemantics(
                domain="generic",
                family="table_export",
                summary="Write dataframe records to CSV files.",
                tags=("csv", "files"),
            ),
            plugin_cls=ExportCsv,
        ),
        build_plugin_asset(
            plugin_id="export/xlsx",
            semantics=PluginSemantics(
                domain="generic",
                family="table_export",
                summary="Write dataframe records to XLSX workbooks.",
                tags=("xlsx", "files"),
            ),
            plugin_cls=ExportXlsx,
        ),
        build_plugin_asset(
            plugin_id="plot/logic_symmetry",
            semantics=PluginSemantics(
                domain="logic",
                family="geometry_plot",
                summary="Render logic symmetry plots from annotated plate-reader summaries.",
                tags=("logic", "geometry"),
            ),
            plugin_cls=LogicSymmetryPlot,
        ),
        build_plugin_asset(
            plugin_id="transform/crosstalk_pairs",
            semantics=PluginSemantics(
                domain="logic",
                family="summary_transform",
                summary="Derive pairwise logic crosstalk rankings from fold-change tables.",
                tags=("logic", "pairs", "summary"),
            ),
            plugin_cls=CrosstalkPairs,
        ),
        build_plugin_asset(
            plugin_id="transform/sfxi",
            semantics=PluginSemantics(
                domain="logic",
                family="summary_transform",
                summary="Compute SFXI vec8 logic summaries from annotated plate-reader traces.",
                tags=("logic", "summary", "sfxi"),
            ),
            plugin_cls=SFXITransform,
        ),
        build_plugin_asset(
            plugin_id="plot/ts_and_snap",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="composite_plot",
                summary="Render paired time-series and snapshot summary panels.",
                tags=("time_series", "snapshot"),
            ),
            plugin_cls=TSAndSnapPlot,
        ),
        build_plugin_asset(
            plugin_id="validator/to_tidy_plus_map",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="contract_promotion",
                summary="Promote tidy tables to annotated plate-reader contracts when metadata is present.",
                tags=("contract", "annotation"),
            ),
            plugin_cls=PromoteToTidyPlusMap,
        ),
        build_plugin_asset(
            plugin_id="transform/ratio",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="derived_channel",
                summary="Create derived ratio channels from aligned tidy measurements.",
                tags=("ratios", "derived_signal"),
            ),
            plugin_cls=RatioTransform,
        ),
        build_plugin_asset(
            plugin_id="transform/retron_sponge_metrics",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="summary_transform",
                summary="Compute matched-control sponge-screen kinetics and ranking summaries from annotated traces.",
                tags=("sponge", "screen", "matched_control", "summary"),
            ),
            plugin_cls=RetronSpongeMetrics,
        ),
        build_plugin_asset(
            plugin_id="plot/distributions",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="distribution",
                summary="Render channel-wise distribution plots from tidy measurements.",
                tags=("density", "qc"),
            ),
            plugin_cls=DistributionsPlot,
        ),
        build_plugin_asset(
            plugin_id="transform/sample_map",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="metadata_merge",
                summary="Attach well-position sample maps to tidy plate-reader traces.",
                tags=("well_map", "annotation"),
            ),
            plugin_cls=SampleMapMerge,
        ),
        build_plugin_asset(
            plugin_id="transform/sample_metadata",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="metadata_merge",
                summary="Attach sample-keyed metadata tables to tidy plate-reader rows.",
                tags=("annotation", "table_join"),
            ),
            plugin_cls=SampleMetadataMerge,
        ),
        build_plugin_asset(
            plugin_id="transform/overflow_handling",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="quality_filter",
                summary="Mask, drop, or cap overflowed plate-reader measurements.",
                tags=("overflow", "qc"),
            ),
            plugin_cls=OverflowHandling,
        ),
        build_plugin_asset(
            plugin_id="transform/blank_correction",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="signal_correction",
                summary="Detect and optionally subtract blank control wells.",
                tags=("blanks", "normalization"),
            ),
            plugin_cls=BlankCorrection,
        ),
        build_plugin_asset(
            plugin_id="plot/snapshot_barplot",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="snapshot_bar",
                summary="Render grouped snapshot barplots at a selected timepoint.",
                tags=("snapshot", "bars"),
            ),
            plugin_cls=SnapshotBarplot,
        ),
        build_plugin_asset(
            plugin_id="plot/snapshot_heatmap",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="snapshot_heatmap",
                summary="Render heatmaps for snapshot or fold-change plate-reader summaries.",
                tags=("snapshot", "heatmap"),
            ),
            plugin_cls=SnapshotHeatmapPlot,
        ),
        build_plugin_asset(
            plugin_id="transform/fold_change",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="summary_transform",
                summary="Summarize nearest-time fold-change tables from tidy signals.",
                tags=("fold_change", "snapshot_summary"),
            ),
            plugin_cls=FoldChange,
        ),
        build_plugin_asset(
            plugin_id="plot/time_series",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="time_series",
                summary="Render grouped time-series plots from tidy plate-reader traces.",
                tags=("kinetics", "channels"),
            ),
            plugin_cls=TimeSeriesPlot,
        ),
        build_plugin_asset(
            plugin_id="ingest/synergy_h1",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="workbook_ingest",
                summary="Parse Synergy H1 workbooks into tidy plate-reader traces.",
                tags=("xlsx", "kinetic", "snapshot"),
            ),
            plugin_cls=SynergyH1,
        ),
    ]
)


def builtin_plugin_asset_catalog(*, categories: set[str] | None = None) -> AssetCatalog:
    if categories is None:
        return _BUILTIN_PLUGIN_CATALOG
    return AssetCatalog(
        [descriptor for descriptor in _BUILTIN_PLUGIN_CATALOG.all() if descriptor.category in categories]
    )


def builtin_plugin_descriptors(*, categories: set[str] | None = None) -> tuple[AssetDescriptor, ...]:
    return builtin_plugin_asset_catalog(categories=categories).all()
