from __future__ import annotations

from reader.plugins.validator.to_tidy_plus_map import PromoteToTidyPlusMap
from reader.workbench.assets import AssetDescriptor, build_plugin_asset
from reader.workbench.ontology import PluginSemantics

BUILTIN_PLUGIN_DESCRIPTORS: tuple[AssetDescriptor, ...] = (
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
)
