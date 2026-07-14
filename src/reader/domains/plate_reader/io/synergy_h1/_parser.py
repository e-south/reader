from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ._kinetic import tidy_kinetic_blocks
from ._shared import (
    ensure_excel_path,
    extract_sheet_datetime,
    is_blank_measurement,
    normalize_channel_map,
    normalize_time_series,
    require,
    require_unique_measurements,
    resolve_channel,
)
from ._snapshot import tidy_snapshot_block


def _selected_sheets(excel: pd.ExcelFile, sheet_names: Sequence[str] | None) -> list[str]:
    sheets = list(sheet_names or excel.sheet_names)
    require(sheets, "Workbook has no sheets")
    for sheet in sheets:
        require(sheet in excel.sheet_names, f"Sheet {sheet!r} not found in workbook")
    return sheets


def _elapsed_hours_by_sheet(excel: pd.ExcelFile, sheets: Sequence[str]) -> dict[str, float]:
    datetimes = {sheet: extract_sheet_datetime(excel, sheet) for sheet in sheets}
    first_datetime = min(datetimes.values())
    return {sheet: (value - first_datetime).total_seconds() / 3600.0 for sheet, value in datetimes.items()}


def _split_snapshot_vs_kinetic(
    frame: pd.DataFrame,
    *,
    channels: Sequence[str] | None,
    channel_map_ci: Mapping[str, str],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    def row_has(index: int, keyword: str) -> bool:
        return frame.iloc[index].astype(str).str.contains(keyword, case=False, na=False).any()

    results_index = next((index for index in frame.index if row_has(index, "Results")), None)
    if results_index is None:
        return None, None

    start = results_index + 1
    for index in frame.index[start:]:
        if frame.iloc[index].dropna(how="all").empty:
            start = index + 1
            break
    time_index = next((index for index in frame.index[start:] if row_has(index, "Time")), None)

    def starts_kinetic_block(index: int) -> bool:
        first_cell = frame.iat[index, 0]
        if is_blank_measurement(first_cell):
            return False
        return resolve_channel(str(first_cell), channels=channels, channel_map_ci=channel_map_ci) is not None

    channel_index = next((index for index in frame.index[start:] if starts_kinetic_block(index)), None)
    boundaries = [index for index in (time_index, channel_index) if index is not None]
    cut = min(boundaries) if boundaries else None
    snapshot = (
        frame.iloc[start:cut].reset_index(drop=True) if cut is not None else frame.iloc[start:].reset_index(drop=True)
    )
    kinetic = frame.iloc[cut:].reset_index(drop=True) if cut is not None else None
    return snapshot, kinetic


def _find_kinetic_section(
    frame: pd.DataFrame,
    *,
    sheet_name: str,
    channels: Sequence[str] | None,
    channel_map_ci: Mapping[str, str],
) -> pd.DataFrame | None:
    for index in frame.index:
        cell = str(frame.iat[index, 0]).strip()
        if not cell:
            continue
        try:
            resolved = resolve_channel(cell, channels=channels, channel_map_ci=channel_map_ci)
        except ValueError as error:
            raise ValueError(f"{error} in kinetic sheet {sheet_name!r}") from error
        if resolved is not None:
            return frame.iloc[index:].reset_index(drop=True)
    return None


def _finalize_measurements(
    frame: pd.DataFrame,
    *,
    channels: Sequence[str] | None,
    expected_channels: set[str],
    missing_message: str,
    time_round_decimals: int | None,
    time_step_h: float | None,
    filter_to_channels: bool,
) -> pd.DataFrame:
    result = frame
    if filter_to_channels and channels:
        result = result[result["channel"].isin(channels)].reset_index(drop=True)

    result["time"] = normalize_time_series(
        result["time"],
        time_round_decimals=time_round_decimals,
        time_step_h=time_step_h,
    )
    result["value"] = pd.to_numeric(result["value"], errors="raise")
    require(result["time"].ge(0).all(), "Internal error: negative time encountered after alignment")
    require(result["time"].notna().all(), "Internal error: time contains NaN after alignment")

    missing = expected_channels - set(result["channel"].astype(str).unique())
    require(not missing, f"{missing_message}: {sorted(missing)}")

    result["position"] = result["position"].astype(str)
    result["channel"] = result["channel"].astype(str)
    require_unique_measurements(result)
    return result.reset_index(drop=True)


def parse_snapshot_and_timeseries(
    path: str | Path,
    *,
    channels: Sequence[str] | None = None,
    channel_map: Mapping[str, str] | None = None,
    sheet_names: Sequence[str] | None = None,
    time_round_decimals: int | None = 12,
    time_step_h: float | None = None,
    include_snapshot: bool = True,
    include_kinetic: bool = True,
) -> pd.DataFrame:
    """Parse snapshot and kinetic blocks from one Synergy H1 workbook."""
    workbook = Path(path)
    ensure_excel_path(workbook)
    channel_map_ci = normalize_channel_map(channel_map)
    require(channels or channel_map_ci, "Provide either 'channels' or 'channel_map'")

    frames: list[pd.DataFrame] = []
    with pd.ExcelFile(workbook) as excel:
        sheets = _selected_sheets(excel, sheet_names)
        elapsed_by_sheet = _elapsed_hours_by_sheet(excel, sheets)
        for sheet_index, sheet in enumerate(sheets):
            raw = excel.parse(sheet_name=sheet, header=None, dtype=str)
            snapshot, kinetic = _split_snapshot_vs_kinetic(
                raw,
                channels=channels,
                channel_map_ci=channel_map_ci,
            )

            if include_snapshot and snapshot is not None and not snapshot.empty:
                frames.append(
                    tidy_snapshot_block(
                        snapshot,
                        elapsed_h=elapsed_by_sheet[sheet],
                        sheet_index=sheet_index,
                        sheet_name=sheet,
                        channels=channels,
                        channel_map_ci=channel_map_ci,
                    )
                )
                declared_channels = sorted(set(channels or channel_map_ci.values()))
                logging.getLogger("reader").debug(
                    "channel normalization (snapshot, sheet %s): declared=%s",
                    sheet,
                    declared_channels,
                )
            if include_kinetic and kinetic is not None and not kinetic.empty:
                frames.append(
                    tidy_kinetic_blocks(
                        kinetic,
                        elapsed_h=elapsed_by_sheet[sheet],
                        sheet_index=sheet_index,
                        sheet_name=sheet,
                        channels=channels,
                        channel_map_ci=channel_map_ci,
                    )
                )

    require(frames, f"No parsable data found in {workbook.name}")
    result = pd.concat(frames, ignore_index=True)

    expected_channels = set(channels or channel_map_ci.values())
    requested_sources = {
        source for source, requested in (("snapshot", include_snapshot), ("kinetic", include_kinetic)) if requested
    }
    sheet_has_snapshot = result.groupby("sheet_index")["source"].transform(lambda values: (values == "snapshot").any())
    sheet_min_time = result.groupby("sheet_index")["time"].transform("min")
    overlapping_initial_kinetic = (
        (result["source"] == "kinetic") & sheet_has_snapshot & (result["time"] == sheet_min_time)
    )
    result = result.loc[~overlapping_initial_kinetic]

    observed_sources = set(result["source"].astype(str).unique())
    missing_sources = requested_sources - observed_sources
    require(not missing_sources, f"Missing requested Synergy data sources: {sorted(missing_sources)}")
    for source in sorted(requested_sources):
        observed_channels = set(result.loc[result["source"] == source, "channel"].astype(str).unique())
        missing_source_channels = expected_channels - observed_channels
        require(
            not missing_source_channels,
            f"{source.title()} data missing for channels: {sorted(missing_source_channels)}",
        )

    return _finalize_measurements(
        result,
        channels=channels,
        expected_channels=expected_channels,
        missing_message="Missing data for channels",
        time_round_decimals=time_round_decimals,
        time_step_h=time_step_h,
        filter_to_channels=True,
    )


def parse_kinetic_only(
    path: str | Path,
    *,
    channels: Sequence[str] | None = None,
    channel_map: Mapping[str, str] | None = None,
    sheet_names: Sequence[str] | None = None,
    time_round_decimals: int | None = 12,
    time_step_h: float | None = None,
) -> pd.DataFrame:
    """Parse kinetic blocks from one Synergy H1 workbook."""
    workbook = Path(path)
    ensure_excel_path(workbook)
    channel_map_ci = normalize_channel_map(channel_map)
    require(channels or channel_map_ci, "Provide either 'channels' or 'channel_map'")

    frames: list[pd.DataFrame] = []
    with pd.ExcelFile(workbook) as excel:
        sheets = _selected_sheets(excel, sheet_names)
        elapsed_by_sheet = _elapsed_hours_by_sheet(excel, sheets)
        for sheet_index, sheet in enumerate(sheets):
            raw = excel.parse(sheet_name=sheet, header=None, dtype=str)
            kinetic = _find_kinetic_section(
                raw,
                sheet_name=sheet,
                channels=channels,
                channel_map_ci=channel_map_ci,
            )
            require(kinetic is not None, f"No kinetic data found in sheet {sheet!r}")
            frames.append(
                tidy_kinetic_blocks(
                    kinetic,
                    elapsed_h=elapsed_by_sheet[sheet],
                    sheet_index=sheet_index,
                    sheet_name=sheet,
                    channels=channels,
                    channel_map_ci=channel_map_ci,
                )
            )

    require(frames, f"No kinetic readings found in {workbook.name}")
    expected_channels = {str(channel) for channel in (channels or channel_map_ci.values())}
    return _finalize_measurements(
        pd.concat(frames, ignore_index=True),
        channels=channels,
        expected_channels=expected_channels,
        missing_message="Kinetic data missing for channels",
        time_round_decimals=time_round_decimals,
        time_step_h=time_step_h,
        filter_to_channels=False,
    )
