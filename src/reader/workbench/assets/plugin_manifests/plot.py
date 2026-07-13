from __future__ import annotations

from reader.plugins.plot.distributions import DistributionsPlot
from reader.plugins.plot.logic_symmetry import LogicSymmetryPlot
from reader.plugins.plot.retron_summary import RetronSummaryPlot
from reader.plugins.plot.retron_trace import RetronTracePlot
from reader.plugins.plot.sfxi_setpoint_scatter import SFXISetpointScatterPlot
from reader.plugins.plot.sfxi_triptych_sequence import SFXITriptychSequencePlot
from reader.plugins.plot.sfxi_vec8_heatmap import SFXIVec8HeatmapPlot
from reader.plugins.plot.snapshot_barplot import SnapshotBarplot
from reader.plugins.plot.snapshot_heatmap import SnapshotHeatmapPlot
from reader.plugins.plot.time_series import TimeSeriesPlot
from reader.plugins.plot.ts_and_snap import TSAndSnapPlot
from reader.workbench.ontology import PluginSemantics

from ..types import AssetDescriptor, build_plugin_asset

BUILTIN_PLUGIN_DESCRIPTORS: tuple[AssetDescriptor, ...] = (
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
        plugin_id="plot/sfxi_setpoint_scatter",
        semantics=PluginSemantics(
            domain="logic",
            family="sfxi_objective_scatter",
            summary="Render OPAL-compatible SFXI setpoint scatter plots over logic_fidelity and effect_scaled.",
            tags=("logic", "sfxi", "setpoint", "scatter"),
        ),
        plugin_cls=SFXISetpointScatterPlot,
    ),
    build_plugin_asset(
        plugin_id="plot/sfxi_triptych_sequence",
        semantics=PluginSemantics(
            domain="logic",
            family="sfxi_triptych_sequence",
            summary="Render SFXI promoter kinetics, snapshot, and sequence annotation figure bundles.",
            tags=("logic", "sfxi", "triptych", "sequence", "baserender"),
        ),
        plugin_cls=SFXITriptychSequencePlot,
    ),
    build_plugin_asset(
        plugin_id="plot/sfxi_vec8_heatmap",
        semantics=PluginSemantics(
            domain="logic",
            family="sfxi_vec8_heatmap",
            summary="Render a heatmap over one experiment's SFXI vec8 logic-shape and reference-normalized intensity channels.",
            tags=("logic", "sfxi", "vec8", "heatmap"),
        ),
        plugin_cls=SFXIVec8HeatmapPlot,
    ),
    build_plugin_asset(
        plugin_id="plot/retron_trace",
        semantics=PluginSemantics(
            domain="plate_reader",
            family="matched_control_kinetics",
            summary="Render retron sponge trace metrics such as burden, baseline shifts, matched-control traces, and induced effects.",
            tags=("retron", "sponge", "kinetics", "matched_control"),
        ),
        plugin_cls=RetronTracePlot,
    ),
    build_plugin_asset(
        plugin_id="plot/retron_summary",
        semantics=PluginSemantics(
            domain="plate_reader",
            family="matched_control_summary",
            summary="Render retron sponge interaction, heatmap, stress-modulation, and Pareto summary figures.",
            tags=("retron", "sponge", "summary", "ranking"),
        ),
        plugin_cls=RetronSummaryPlot,
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
        plugin_id="plot/time_series",
        semantics=PluginSemantics(
            domain="plate_reader",
            family="time_series",
            summary="Render grouped time-series plots from tidy plate-reader traces.",
            tags=("kinetics", "channels"),
        ),
        plugin_cls=TimeSeriesPlot,
    ),
)
