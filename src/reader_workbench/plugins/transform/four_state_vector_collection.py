from __future__ import annotations

from reader_workbench.domains.logic.four_state_vector.collection import (
    FourStateVectorSource,
    collect_four_state_vector_sources,
)
from reader_workbench.workbench.ports import dataframe_output, record_collection_input
from reader_workbench.workbench.records import SourceRecordCollection
from reader_workbench.workbench.registry import Plugin, PluginConfig


class FourStateVectorCollectionCfg(PluginConfig):
    pass


class FourStateVectorCollectionTransform(Plugin):
    """Collect exact vector revisions; workspace discovery stays in Reader core."""

    ConfigModel = FourStateVectorCollectionCfg

    @classmethod
    def input_ports(cls):
        return {"sources": record_collection_input("sources", "logic.four_state_vector.v1")}

    @classmethod
    def output_ports(cls):
        return {"vectors": dataframe_output("vectors", "logic.four_state_vector_collection.v1")}

    def run(self, ctx, inputs, cfg):
        collection: SourceRecordCollection = inputs["sources"]
        sources = tuple(
            FourStateVectorSource(
                resource_id=item.ref.resource_id,
                experiment_id=item.ref.experiment_id,
                record_id=item.ref.record_id,
                revision_digest=item.revision_digest,
                frame=item.load_dataframe(),
            )
            for item in collection
        )
        return {"vectors": collect_four_state_vector_sources(sources).frame}
