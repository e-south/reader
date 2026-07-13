from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import pandas as pd

from ._shared import (
    canonical_channel,
    coerce_measurements,
    is_blank_measurement,
    require,
    resolve_channel,
    time_column,
    well_headers,
)


def tidy_kinetic_blocks(
    kinetic: pd.DataFrame,
    *,
    elapsed_h: float,
    sheet_index: int,
    sheet_name: str,
    channels: Sequence[str] | None,
    channel_map_ci: Mapping[str, str],
) -> pd.DataFrame:
    logger = logging.getLogger("reader")
    raw_to_resolved: dict[str, str] = {}
    raw_to_canonical: dict[str, str] = {}

    def row_has(frame: pd.DataFrame, index: int, pattern: str) -> bool:
        return frame.iloc[index].astype(str).str.contains(pattern, case=False, na=False).any()

    def looks_like_label(first_cell: object) -> bool:
        value = str(first_cell or "").strip()
        if not value or value.lower() == "nan":
            return False
        if ":" in value:
            return True
        try:
            return resolve_channel(value, channels=channels, channel_map_ci=channel_map_ci) is not None
        except ValueError as error:
            raise ValueError(f"{error} in kinetic sheet {sheet_name!r}") from error

    label_rows = [index for index, row in kinetic.iterrows() if looks_like_label(row.iat[0])]
    require(
        label_rows,
        "No kinetic blocks found: expected a channel label in column A "
        "(e.g., 'OD600' or 'OD600: …') preceding a 'Time' header row.",
    )

    parts: list[pd.DataFrame] = []
    for block_index, start in enumerate(label_rows):
        end = label_rows[block_index + 1] if block_index + 1 < len(label_rows) else len(kinetic)
        block = kinetic.iloc[start:end].reset_index(drop=True)
        section_end = next((index for index in range(1, len(block)) if row_has(block, index, r"^Results$")), None)
        if section_end is not None:
            block = block.iloc[:section_end].reset_index(drop=True)

        raw_channel = str(block.iat[0, 0]).strip()
        canonical = canonical_channel(raw_channel)
        channel = resolve_channel(raw_channel, channels=channels, channel_map_ci=channel_map_ci)
        if channel is None:
            continue

        raw_to_canonical.setdefault(raw_channel, canonical)
        previous = raw_to_resolved.get(raw_channel)
        if previous is not None and previous != channel:
            logger.warning(
                "[warn]inconsistent channel resolution[/warn] • raw=%r canon=%r previously→%r now→%r",
                raw_channel,
                canonical,
                previous,
                channel,
            )
        raw_to_resolved[raw_channel] = channel

        header_index = next((index for index in range(1, len(block)) if row_has(block, index, r"^Time")), None)
        require(header_index is not None, f"Time header not found in kinetic block for channel {channel!r}")
        header = block.iloc[header_index].astype(str).tolist()
        time_key = time_column(header)
        wells = well_headers(header)
        require(wells, f"No well columns (A1..H12) in kinetic header for {channel!r}")

        data = block.iloc[header_index + 1 :].reset_index(drop=True)
        data.columns = header
        raw_times = data[time_key]
        parsed_times = pd.to_timedelta(raw_times, errors="coerce")
        has_measurement = data[wells].map(lambda value: not is_blank_measurement(value)).any(axis=1)
        invalid_time = parsed_times.isna() & (
            raw_times.map(lambda value: not is_blank_measurement(value)) | has_measurement
        )
        if invalid_time.any():
            index = invalid_time[invalid_time].index[0]
            token = raw_times.loc[index]
            raise ValueError(f"Invalid kinetic time token {token!r} in sheet {sheet_name!r}, channel {channel!r}")
        time_hours = parsed_times.dt.total_seconds() / 3600.0
        data = data.assign(__time_hr=time_hours).loc[lambda frame: frame["__time_hr"].notna()].reset_index(drop=True)
        require(not data.empty, f"Non-parsable time values in kinetic block for {channel!r}")

        first_relative_time = float(data["__time_hr"].iloc[0])
        data["__time_hr"] = float(elapsed_h) + (data["__time_hr"] - first_relative_time)

        melted = data.melt(
            id_vars=["__time_hr"],
            value_vars=wells,
            var_name="position",
            value_name="value",
        ).rename(columns={"__time_hr": "time"})
        melted["channel"] = channel
        melted["sheet_index"] = sheet_index
        melted["sheet_name"] = sheet_name
        melted["source"] = "kinetic"
        parts.append(melted)

    require(parts, f"No configured kinetic channel blocks found in sheet {sheet_name!r}")
    result = coerce_measurements(pd.concat(parts, ignore_index=True), source="kinetic")

    if raw_to_resolved:
        pairs = [
            f"{raw!r} → {raw_to_canonical.get(raw, '?')!r} → {raw_to_resolved[raw]!r}"
            for raw in sorted(raw_to_resolved)
        ]
        logger.debug("channel normalization (kinetic, sheet %s): %s", sheet_name, "; ".join(pairs))

    return result
