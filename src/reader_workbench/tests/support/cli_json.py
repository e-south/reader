from __future__ import annotations

import json


def success_data(output: str) -> dict[str, object]:
    envelope = json.loads(output)
    assert envelope["schema"] == "reader.cli/v1"
    assert envelope["ok"] is True
    assert envelope["error"] is None
    return envelope["data"]


def error_data(output: str) -> dict[str, object]:
    envelope = json.loads(output)
    assert envelope["schema"] == "reader.cli/v1"
    assert envelope["ok"] is False
    assert envelope["data"] is None
    return envelope["error"]
