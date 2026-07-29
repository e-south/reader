from __future__ import annotations

from pathlib import Path

from reader.domains.logic.sfxi.vec8_aggregate import LoadedSFXIVec8Source, aggregate_sfxi_vec8_sources
from reader.workbench.ports import dataframe_output, record_collection_input
from reader.workbench.records import SourceRecordCollection
from reader.workbench.registry import Plugin, PluginConfig


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
        return {"vec8": dataframe_output("vec8", "sfxi.vec8_collection.v1")}

    def run(self, ctx, inputs, cfg):
        collection: SourceRecordCollection = inputs["sources"]
        loaded = tuple(
            LoadedSFXIVec8Source(
                source_id=item.ref.experiment_id,
                source_path=Path(item.ref.experiment_id),
                table_path=Path(item.ref.record_id),
                source_kind="experiment_record",
                frame=item.load_dataframe(),
                record_id=item.ref.record_id,
            )
            for item in collection
        )
        frame = aggregate_sfxi_vec8_sources(loaded).frame.drop(
            columns=["source_path", "table_path", "source_kind"],
            errors="ignore",
        )
        return {"vec8": frame}
