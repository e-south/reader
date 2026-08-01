from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from reader_workbench.domains.logic.four_state_vector.collection import (
    FourStateVectorSource,
    collect_four_state_vector_sources,
)
from reader_workbench.domains.logic.four_state_vector.collection.render import _display_row_labels, _ordered_plot_frame
from reader_workbench.errors import FourStateVectorError


def _revision_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _vector(*design_ids: str) -> pd.DataFrame:
    row_count = len(design_ids)
    return pd.DataFrame(
        {
            "design_id": list(design_ids),
            "reference_design_id": ["reference"] * row_count,
            "time_selected_h": [12.0] * row_count,
            "intensity_log2_offset_delta": [0.0] * row_count,
            "r_logic": [4.0] * row_count,
            "v00": [0.0] * row_count,
            "v10": [0.2] * row_count,
            "v01": [0.7] * row_count,
            "v11": [1.0] * row_count,
            "y00_star": [-1.0] * row_count,
            "y10_star": [0.0] * row_count,
            "y01_star": [1.0] * row_count,
            "y11_star": [2.0] * row_count,
            "flat_logic": [False] * row_count,
        }
    )


def test_vector_collection_accepts_only_canonical_in_memory_source_metadata() -> None:
    collection = collect_four_state_vector_sources(
        (
            FourStateVectorSource(
                resource_id="second",
                experiment_id="experiment-2",
                revision_digest=_revision_digest("revision-2"),
                frame=_vector("design-2"),
                record_id="four_state_vector/vector",
            ),
        )
    )

    assert collection.frame["source_resource_id"].tolist() == ["second"]
    assert collection.frame["source_experiment_id"].tolist() == ["experiment-2"]
    assert collection.frame["source_record_id"].tolist() == ["four_state_vector/vector"]
    assert collection.frame["source_record_revision_digest"].tolist() == [_revision_digest("revision-2")]
    assert {"source_path", "table_path", "source_kind"}.isdisjoint(collection.frame.columns)
    assert not hasattr(collection, "load_four_state_vector_table")


def test_vector_collection_renderer_uses_generic_natural_order_and_labels() -> None:
    collection = collect_four_state_vector_sources(
        (
            FourStateVectorSource(
                resource_id="tenth",
                experiment_id="experiment-10",
                revision_digest=_revision_digest("revision-10"),
                frame=_vector("design-10", "design-2"),
                record_id="vector/ten",
            ),
            FourStateVectorSource(
                resource_id="second",
                experiment_id="experiment-2",
                revision_digest=_revision_digest("revision-2"),
                frame=_vector("design-10", "design-2"),
                record_id="vector/two",
            ),
        )
    )

    ordered = _ordered_plot_frame(collection.frame)

    assert list(zip(ordered["design_id"], ordered["source_resource_id"], strict=True)) == [
        ("design-2", "second"),
        ("design-2", "tenth"),
        ("design-10", "second"),
        ("design-10", "tenth"),
    ]
    assert _display_row_labels(ordered) == [
        "second t=12.00h :: design-2",
        "tenth t=12.00h :: design-2",
        "second t=12.00h :: design-10",
        "tenth t=12.00h :: design-10",
    ]


def test_vector_collection_distinguishes_records_from_the_same_experiment() -> None:
    collection = collect_four_state_vector_sources(
        (
            FourStateVectorSource(
                resource_id="first",
                experiment_id="shared-experiment",
                record_id="vector/first",
                revision_digest=_revision_digest("revision-first"),
                frame=_vector("design"),
            ),
            FourStateVectorSource(
                resource_id="second",
                experiment_id="shared-experiment",
                record_id="vector/second",
                revision_digest=_revision_digest("revision-second"),
                frame=_vector("design"),
            ),
        )
    )

    assert collection.frame[["source_resource_id", "source_experiment_id", "source_record_id", "design_id"]].to_dict(
        orient="records"
    ) == [
        {
            "source_resource_id": "first",
            "source_experiment_id": "shared-experiment",
            "source_record_id": "vector/first",
            "design_id": "design",
        },
        {
            "source_resource_id": "second",
            "source_experiment_id": "shared-experiment",
            "source_record_id": "vector/second",
            "design_id": "design",
        },
    ]


def test_vector_collection_rejects_duplicate_exact_source_records() -> None:
    with pytest.raises(FourStateVectorError, match="source record identities must be unique"):
        collect_four_state_vector_sources(
            (
                FourStateVectorSource(
                    resource_id="first-alias",
                    experiment_id="shared-experiment",
                    record_id="vector/shared",
                    revision_digest=_revision_digest("revision-shared"),
                    frame=_vector("first-design"),
                ),
                FourStateVectorSource(
                    resource_id="second-alias",
                    experiment_id="shared-experiment",
                    record_id="vector/shared",
                    revision_digest=_revision_digest("revision-shared"),
                    frame=_vector("second-design"),
                ),
            )
        )


@pytest.mark.parametrize(
    "revision_digest",
    (
        "sha256:revision",
        "sha256:" + "a" * 63,
        "sha256:" + "a" * 65,
        "sha256:" + "A" * 64,
        "sha512:" + "a" * 64,
    ),
)
def test_vector_collection_rejects_noncanonical_revision_digest(revision_digest: str) -> None:
    with pytest.raises(FourStateVectorError, match="revision_digest must be a canonical sha256 digest"):
        collect_four_state_vector_sources(
            (
                FourStateVectorSource(
                    resource_id="source",
                    experiment_id="experiment",
                    record_id="four_state_vector/vector",
                    revision_digest=revision_digest,
                    frame=_vector("design"),
                ),
            )
        )
