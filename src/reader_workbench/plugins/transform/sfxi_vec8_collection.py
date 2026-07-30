from __future__ import annotations

from reader_workbench.domains.logic.sfxi.vec8_aggregate import SFXIVec8Source, aggregate_sfxi_vec8_sources
from reader_workbench.workbench.ports import dataframe_output, record_collection_input
from reader_workbench.workbench.records import SourceRecordCollection
from reader_workbench.workbench.registry import Plugin, PluginConfig


class SFXIVec8CollectionCfg(PluginConfig):
    pass


class SFXIVec8CollectionTransform(Plugin):
    """Combine exact vec8 record revisions; workspace discovery stays in Reader core."""

    ConfigModel = SFXIVec8CollectionCfg

    @classmethod
    def input_ports(cls):
        return {"sources": record_collection_input("sources", "sfxi.vec8.v3")}

    @classmethod
    def output_ports(cls):
        return {"vec8": dataframe_output("vec8", "sfxi.vec8_collection.v2")}

    def run(self, ctx, inputs, cfg):
        collection: SourceRecordCollection = inputs["sources"]
        sources = tuple(
            SFXIVec8Source(
                resource_id=item.ref.resource_id,
                experiment_id=item.ref.experiment_id,
                record_id=item.ref.record_id,
                revision_digest=item.revision_digest,
                frame=item.load_dataframe(),
            )
            for item in collection
        )
        return {"vec8": aggregate_sfxi_vec8_sources(sources).frame}
