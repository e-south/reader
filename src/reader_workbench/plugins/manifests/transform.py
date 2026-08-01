from __future__ import annotations

from reader_workbench.plugins.transform.alias import AliasTransform
from reader_workbench.plugins.transform.assay_labels import AnnotationLabelsTransform
from reader_workbench.plugins.transform.blank import BlankCorrection
from reader_workbench.plugins.transform.crosstalk_pairs import CrosstalkPairs
from reader_workbench.plugins.transform.cytometry_gating import CytometryGatingTransform
from reader_workbench.plugins.transform.fold_change import FoldChange
from reader_workbench.plugins.transform.four_state_event_window import FourStateEventWindowTransform
from reader_workbench.plugins.transform.four_state_vector import FourStateVectorTransform
from reader_workbench.plugins.transform.four_state_vector_collection import FourStateVectorCollectionTransform
from reader_workbench.plugins.transform.logic_symmetry import LogicSymmetryTransform
from reader_workbench.plugins.transform.outlier_filter import OutlierFilter
from reader_workbench.plugins.transform.overflow import OverflowHandling
from reader_workbench.plugins.transform.ratio import RatioTransform
from reader_workbench.plugins.transform.sample_map import SampleMapMerge
from reader_workbench.plugins.transform.sample_metadata import SampleMetadataMerge
from reader_workbench.workbench.assets import AssetDescriptor, build_plugin_asset
from reader_workbench.workbench.ontology import PluginSemantics

BUILTIN_PLUGIN_DESCRIPTORS: tuple[AssetDescriptor, ...] = (
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
        plugin_id="transform/cytometry_gating",
        semantics=PluginSemantics(
            domain="cytometry",
            family="gating",
            summary="Apply an explicit cytometry gating, threshold, grouping, and QC policy.",
            tags=("gating", "threshold", "qc"),
        ),
        plugin_cls=CytometryGatingTransform,
    ),
    build_plugin_asset(
        plugin_id="transform/logic_symmetry",
        semantics=PluginSemantics(
            domain="logic",
            family="summary_transform",
            summary="Compute logic-symmetry metrics over an explicit four-state mapping.",
            tags=("logic", "geometry", "summary"),
        ),
        plugin_cls=LogicSymmetryTransform,
    ),
    build_plugin_asset(
        plugin_id="transform/four_state_vector",
        semantics=PluginSemantics(
            domain="logic",
            family="summary_transform",
            summary="Compute four-state vector logic summaries from annotated plate-reader traces.",
            tags=("logic", "summary", "four_state_vector"),
        ),
        plugin_cls=FourStateVectorTransform,
    ),
    build_plugin_asset(
        plugin_id="transform/four_state_vector_collection",
        semantics=PluginSemantics(
            domain="logic",
            family="record_collection",
            summary="Collect exact four-state vector record revisions from multiple Reader experiments.",
            tags=("logic", "collection", "provenance"),
        ),
        plugin_cls=FourStateVectorCollectionTransform,
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
        plugin_id="transform/four_state_event_window",
        semantics=PluginSemantics(
            domain="plate_reader",
            family="event_relative_summary",
            summary="Materialize event-relative summaries from provenance-bound source records.",
            tags=("event", "window", "aggregate", "provenance"),
        ),
        plugin_cls=FourStateEventWindowTransform,
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
            domain="generic",
            family="metadata_merge",
            summary="Attach sample-keyed metadata tables to tidy measurement rows.",
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
        plugin_id="transform/fold_change",
        semantics=PluginSemantics(
            domain="plate_reader",
            family="summary_transform",
            summary="Summarize nearest-time fold-change tables from tidy signals.",
            tags=("fold_change", "snapshot_summary"),
        ),
        plugin_cls=FoldChange,
    ),
)
