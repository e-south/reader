from __future__ import annotations

from pathlib import Path

import pytest

from reader.workbench.graph import FileRef, RecordRef, ResourceRef, input_ref_from_dict


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"record": "ingest/df"}, RecordRef(record_id="ingest/df")),
        ({"file": "inputs/raw.xlsx"}, FileRef(path=Path("inputs/raw.xlsx"))),
        (
            {"resource": "sample_map", "path": "inputs/metadata.xlsx"},
            ResourceRef(resource_id="sample_map", path=Path("inputs/metadata.xlsx")),
        ),
    ],
)
def test_input_ref_from_dict_accepts_one_exact_reference_shape(payload, expected) -> None:
    assert input_ref_from_dict(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"record": "ingest/df", "file": "inputs/raw.xlsx"},
        {"record": "ingest/df", "extra": "ignored"},
        {"resource": "sample_map"},
        {"file": ""},
    ],
)
def test_input_ref_from_dict_rejects_ambiguous_or_incomplete_shapes(payload) -> None:
    with pytest.raises(ValueError):
        input_ref_from_dict(payload)
