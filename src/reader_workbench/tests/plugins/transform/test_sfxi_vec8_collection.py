from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from reader_workbench.plugins.plot.sfxi_vec8_collection import SFXIVec8CollectionHeatmapPlot
from reader_workbench.plugins.transform.sfxi_vec8_collection import (
    SFXIVec8CollectionCfg,
    SFXIVec8CollectionTransform,
)


def _vec8() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": ["design"],
            "reference_design_id": ["reference"],
            "intensity_log2_offset_delta": [0.0],
            "r_logic": [4.0],
            "v00": [0.0],
            "v10": [0.2],
            "v01": [0.7],
            "v11": [1.0],
            "y00_star": [-1.0],
            "y10_star": [0.0],
            "y01_star": [1.0],
            "y11_star": [2.0],
            "flat_logic": [False],
        }
    )


def _revision_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _ResolvedSource:
    ref: SimpleNamespace
    revision_digest: str

    def load_dataframe(self) -> pd.DataFrame:
        return _vec8()


def test_collection_transform_preserves_exact_source_record_identity() -> None:
    source = _ResolvedSource(
        ref=SimpleNamespace(
            resource_id="candidate_a",
            experiment_id="experiment-a",
            record_id="sfxi_vec8/vec8",
        ),
        revision_digest=_revision_digest("exact-revision"),
    )

    result = SFXIVec8CollectionTransform().run(
        None,
        {"sources": (source,)},
        SFXIVec8CollectionCfg(),
    )["vec8"]

    assert result[
        [
            "source_resource_id",
            "source_experiment_id",
            "source_record_id",
            "source_record_revision_digest",
        ]
    ].to_dict(orient="records") == [
        {
            "source_resource_id": "candidate_a",
            "source_experiment_id": "experiment-a",
            "source_record_id": "sfxi_vec8/vec8",
            "source_record_revision_digest": _revision_digest("exact-revision"),
        }
    ]


def test_collection_plugins_share_the_revision_bound_contract() -> None:
    assert SFXIVec8CollectionTransform.output_ports()["vec8"].contract == "sfxi.vec8_collection.v2"
    assert SFXIVec8CollectionHeatmapPlot.input_ports()["vec8"].contract == "sfxi.vec8_collection.v2"
