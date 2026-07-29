from __future__ import annotations

from pathlib import Path

import pytest

from reader.errors import RecordError
from reader.protocols.model import binding_value
from reader.runtime import builtin_runtime
from reader.workbench.cli.helpers import dataframe_record_contracts
from reader.workbench.cli.shared import json_friendly


def test_dataframe_record_contracts_raises_on_corrupt_catalog(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    manifests = outputs / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "records.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(RecordError, match="not valid JSON"):
        dataframe_record_contracts(outputs, runtime=builtin_runtime())


def test_json_friendly_serializes_protocol_binding_value_ref() -> None:
    assert json_friendly(binding_value("sample_map")) == {"binding_value": "sample_map"}
    assert json_friendly(binding_value("sample_map", default="metadata.xlsx")) == {
        "binding_value": "sample_map",
        "default": "metadata.xlsx",
    }
