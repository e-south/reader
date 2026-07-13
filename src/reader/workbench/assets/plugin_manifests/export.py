from __future__ import annotations

from reader.plugins.export.csv import ExportCsv
from reader.plugins.export.xlsx import ExportXlsx
from reader.workbench.ontology import PluginSemantics

from ..types import AssetDescriptor, build_plugin_asset

BUILTIN_PLUGIN_DESCRIPTORS: tuple[AssetDescriptor, ...] = (
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
)
