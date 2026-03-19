from __future__ import annotations

from pathlib import Path

import pytest

from reader.domains.cytometry.io.fcs import parse_fcs_file

pytestmark = pytest.mark.integration


def test_parse_fcs_file_returns_tidy_events_and_channel_metadata() -> None:
    fcs_path = Path("experiments/2026/20260101_cytometer_retron/inputs/retron-26-neg_Data Source - 1.fcs")
    if not fcs_path.exists():
        pytest.skip("Cytometer fixture file missing")

    df, channels = parse_fcs_file(fcs_path, channel_name_field="pns")

    assert {"position", "time", "channel", "value", "sample_id"} <= set(df.columns)
    assert df["sample_id"].nunique() == 1
    assert df["position"].nunique() == 1
    assert {"sample_id", "channel_index", "channel_name"} <= set(channels.columns)
    assert channels["sample_id"].nunique() == 1
