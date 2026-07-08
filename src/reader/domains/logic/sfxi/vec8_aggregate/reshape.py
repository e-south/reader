from __future__ import annotations

import pandas as pd

from .checks import require_normalized_frame
from .constants import METADATA_COLUMNS, VEC8_CHANNELS


def sfxi_vec8_tidy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    require_normalized_frame(frame)
    metadata_columns = [column for column in METADATA_COLUMNS if column in frame.columns]
    tidy = frame.loc[:, [*metadata_columns, *VEC8_CHANNELS]].melt(
        id_vars=metadata_columns,
        value_vars=list(VEC8_CHANNELS),
        var_name="channel",
        value_name="value",
    )
    channel_index = {channel: index for index, channel in enumerate(VEC8_CHANNELS)}
    tidy["channel_index"] = tidy["channel"].map(channel_index).astype(int)
    tidy["value"] = tidy["value"].astype(float)
    sort_columns = [
        column for column in ("source_index", "source_row_index", "channel_index") if column in tidy.columns
    ]
    tidy = tidy.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return tidy.loc[:, [*metadata_columns, "channel", "channel_index", "value"]]
