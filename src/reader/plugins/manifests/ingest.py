from __future__ import annotations

from reader.plugins.ingest.flow_cytometer import FlowCytometerIngest
from reader.plugins.ingest.synergy_h1 import SynergyH1
from reader.workbench.assets import AssetDescriptor, build_plugin_asset
from reader.workbench.ontology import PluginSemantics

BUILTIN_PLUGIN_DESCRIPTORS: tuple[AssetDescriptor, ...] = (
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
        plugin_id="ingest/synergy_h1",
        semantics=PluginSemantics(
            domain="plate_reader",
            family="workbook_ingest",
            summary="Parse Synergy H1 workbooks into tidy plate-reader traces.",
            tags=("xlsx", "kinetic", "snapshot"),
        ),
        plugin_cls=SynergyH1,
    ),
)
