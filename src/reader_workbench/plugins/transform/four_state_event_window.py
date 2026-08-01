from __future__ import annotations

from typing import Any

import pandas as pd

from reader_workbench.domains.plate_reader.analysis.four_state_event_window.contracts import (
    FourStateEventWindowAnalysisSpec,
)
from reader_workbench.domains.plate_reader.analysis.four_state_event_window.materialize import materialize_experiment
from reader_workbench.domains.plate_reader.analysis.four_state_event_window.sources import build_experiment_source
from reader_workbench.workbench.ports import dataframe_output, record_collection_input
from reader_workbench.workbench.records import SourceRecordCollection
from reader_workbench.workbench.registry import Plugin, PluginConfig


class FourStateEventWindowCfg(PluginConfig):
    source: dict[str, Any]
    event: dict[str, Any]
    reductions: list[dict[str, Any]]
    aggregation: dict[str, Any]
    quality: dict[str, Any]


class FourStateEventWindowTransform(Plugin):
    """Materialize event-relative summaries from aligned source record collections."""

    ConfigModel = FourStateEventWindowCfg

    @classmethod
    def input_ports(cls):
        source = "plate_reader.annotated.v1"
        return {
            "response_records": record_collection_input("response_records", source),
            "magnitude_records": record_collection_input("magnitude_records", source),
            "trajectory_records": record_collection_input("trajectory_records", source),
        }

    @classmethod
    def output_ports(cls):
        return {
            "wells": dataframe_output("wells", "plate_reader.four_state_event_window.wells.v3"),
            "designs": dataframe_output("designs", "plate_reader.four_state_event_window.designs.v4"),
            "descriptive_resampling_draws": dataframe_output(
                "descriptive_resampling_draws",
                "plate_reader.four_state_event_window.descriptive_resampling_draws.v3",
            ),
            "traces": dataframe_output("traces", "plate_reader.four_state_event_window.traces.v3"),
            "events": dataframe_output("events", "plate_reader.four_state_event_window.events.v2"),
        }

    def run(self, ctx, inputs, cfg):
        spec = FourStateEventWindowAnalysisSpec.from_mapping(cfg.model_dump())
        response: SourceRecordCollection = inputs["response_records"]
        magnitude: SourceRecordCollection = inputs["magnitude_records"]
        trajectory: SourceRecordCollection = inputs["trajectory_records"]
        experiment_ids = tuple(item.ref.experiment_id for item in response)
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("four-state event-window source experiments must be unique")
        for label, collection in (("magnitude", magnitude), ("trajectory", trajectory)):
            observed = tuple(item.ref.experiment_id for item in collection)
            if observed != experiment_ids:
                raise ValueError(
                    f"four-state event-window {label} source order must match response source experiments: "
                    f"expected {experiment_ids}, got {observed}"
                )

        materialized = []
        for response_item, magnitude_item, trajectory_item in zip(
            response,
            magnitude,
            trajectory,
            strict=True,
        ):
            source = build_experiment_source(
                experiment_id=response_item.ref.experiment_id,
                response_frame=response_item.load_dataframe(),
                magnitude_frame=magnitude_item.load_dataframe(),
                trajectory_frame=trajectory_item.load_dataframe(),
                source_spec=spec.source,
                event_spec=spec.event,
            )
            materialized.append(materialize_experiment(source, request=spec))
        names = ("wells", "designs", "descriptive_resampling_draws", "traces", "events")
        return {
            name: pd.concat([frames[index] for frames in materialized], ignore_index=True)
            for index, name in enumerate(names)
        }
