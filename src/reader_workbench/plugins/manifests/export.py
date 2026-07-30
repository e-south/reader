from __future__ import annotations

from reader_workbench.plugins.export.csv import ExportCsv
from reader_workbench.plugins.export.xlsx import ExportXlsx
from reader_workbench.workbench.assets import AssetDescriptor, build_plugin_asset
from reader_workbench.workbench.ontology import PluginSemantics

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
