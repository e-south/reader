from __future__ import annotations

from reader.protocols.model import binding_value
from reader.workbench.cli.shared import json_friendly


def test_json_friendly_serializes_protocol_binding_value_ref() -> None:
    assert json_friendly(binding_value("sample_map")) == {"binding_value": "sample_map"}
    assert json_friendly(binding_value("sample_map", default="metadata.xlsx")) == {
        "binding_value": "sample_map",
        "default": "metadata.xlsx",
    }
