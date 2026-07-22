from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import pandas as pd

from ._shared import (
    coerce_measurements,
    drop_all_empty_rows,
    is_blank_measurement,
    require,
    resolve_channel_from_map,
)


def tidy_snapshot_block(
    snapshot: pd.DataFrame,
    *,
    elapsed_h: float,
    sheet_index: int,
    sheet_name: str,
    channels: Sequence[str] | None,
    channel_map_ci: Mapping[str, str],
) -> pd.DataFrame:
    require(channel_map_ci, "Snapshot parsing requires an explicit 'channel_map'")
    require(not snapshot.empty, "Empty snapshot block")

    header = snapshot.iloc[0]
    well_columns = [index for index, value in enumerate(header) if str(value).strip().isdigit()]
    require(well_columns, "Snapshot header did not contain numbered well columns")
    well_numbers = [int(str(header[index]).strip()) for index in well_columns]
    invalid_well_numbers = [number for number in well_numbers if number < 1 or number > 12]
    require(
        not invalid_well_numbers,
        f"Snapshot header contains columns outside the 96-well range 1..12: {invalid_well_numbers}",
    )
    duplicate_well_numbers = sorted({number for number in well_numbers if well_numbers.count(number) > 1})
    require(not duplicate_well_numbers, f"Snapshot header contains duplicate well columns: {duplicate_well_numbers}")

    data_rows = drop_all_empty_rows(snapshot.iloc[1:].copy())
    require(not data_rows.empty, "Snapshot block contains a header but no data rows")

    row_column_candidates = [
        column
        for column in range(data_rows.shape[1])
        if data_rows.iloc[:, column].map(lambda value: bool(re.fullmatch(r"[A-H]", str(value).strip()))).any()
    ]
    require(
        len(row_column_candidates) == 1,
        f"Could not identify one row-letter column in snapshot block on sheet {sheet_name!r}",
    )
    row_column = row_column_candidates[0]
    label_columns = [
        column for column in range(data_rows.shape[1]) if column != row_column and column not in well_columns
    ]
    expected_channels = set(channels or channel_map_ci.values())

    def channel_for_data_row(row: pd.Series, *, plate_row: str) -> str:
        declarations: list[tuple[str, str]] = []
        for column in label_columns:
            value = row.iat[column]
            if is_blank_measurement(value):
                continue
            raw_label = str(value).strip()
            try:
                resolved = resolve_channel_from_map(raw_label, channel_map_ci=channel_map_ci)
            except ValueError as error:
                raise ValueError(f"{error} in snapshot sheet {sheet_name!r}, plate row {plate_row!r}") from error
            if resolved is not None:
                declarations.append((raw_label, resolved))
        if not declarations:
            raise ValueError(f"Missing snapshot channel label in sheet {sheet_name!r}, plate row {plate_row!r}")
        if len(declarations) > 1:
            labels = [raw for raw, _ in declarations]
            raise ValueError(
                f"Multiple snapshot channel labels {labels} in sheet {sheet_name!r}, plate row {plate_row!r}"
            )
        return declarations[0][1]

    rows: list[dict[str, object]] = []
    current_plate_row: str | None = None
    current_channels: set[str] = set()

    def finish_plate_row() -> None:
        if current_plate_row is None:
            return
        missing = expected_channels - current_channels
        require(
            not missing,
            f"Missing snapshot channel labels {sorted(missing)} in sheet {sheet_name!r}, "
            f"plate row {current_plate_row!r}",
        )

    for _, data_row in data_rows.iterrows():
        row_marker = data_row.iat[row_column]
        if not is_blank_measurement(row_marker):
            plate_row = str(row_marker).strip()
            require(
                bool(re.fullmatch(r"[A-H]", plate_row)),
                f"Invalid snapshot row label {plate_row!r} in sheet {sheet_name!r}",
            )
            finish_plate_row()
            current_plate_row = plate_row
            current_channels = set()
        require(
            current_plate_row is not None,
            f"Snapshot measurement row precedes a plate-row label in sheet {sheet_name!r}",
        )

        channel = channel_for_data_row(data_row, plate_row=current_plate_row)
        require(
            channel not in current_channels,
            f"Duplicate snapshot channel {channel!r} in sheet {sheet_name!r}, plate row {current_plate_row!r}",
        )
        current_channels.add(channel)
        for well_number, value in zip(well_numbers, data_row.iloc[well_columns].tolist(), strict=True):
            rows.append(
                {
                    "position": f"{current_plate_row}{well_number}",
                    "time": float(elapsed_h),
                    "channel": channel,
                    "value": value,
                    "sheet_index": sheet_index,
                    "sheet_name": sheet_name,
                    "source": "snapshot",
                }
            )
    finish_plate_row()
    return coerce_measurements(pd.DataFrame(rows), source="snapshot")
