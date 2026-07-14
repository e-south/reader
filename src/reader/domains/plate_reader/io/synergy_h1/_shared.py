from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def drop_all_empty_rows(frame: pd.DataFrame) -> pd.DataFrame:
    with pd.option_context("future.no_silent_downcasting", True):
        normalized = frame.replace(r"^\s*$", pd.NA, regex=True)
    return normalized.infer_objects(copy=False).dropna(how="all").reset_index(drop=True)


def ensure_excel_path(path: Path) -> None:
    require(path.exists(), f"Input file not found: {path}")
    require(path.is_file(), f"Synergy H1 input is not a regular file: {path}")
    require(
        path.suffix.lower() == ".xlsx",
        f"Synergy H1 ingest requires a modern .xlsx workbook, got {path.suffix or '<no extension>'!r}",
    )


def require_unique_measurements(frame: pd.DataFrame) -> None:
    keys = ["position", "channel", "time"]
    duplicate_mask = frame.duplicated(subset=keys, keep=False)
    if not duplicate_mask.any():
        return
    preview = frame.loc[duplicate_mask, keys].drop_duplicates().head(8).to_dict(orient="records")
    raise ValueError(f"Synergy measurements must be unique by {keys}; duplicate keys include {preview}")


def probe_synergy_workbook(path: str | Path) -> tuple[str, ...]:
    """Open a Synergy workbook and return its sheet names without parsing sheet data."""
    workbook = Path(path)
    ensure_excel_path(workbook)
    with pd.ExcelFile(workbook) as excel:
        return tuple(str(sheet) for sheet in excel.sheet_names)


def extract_sheet_datetime(excel: pd.ExcelFile, sheet: str) -> datetime:
    metadata = excel.parse(sheet_name=sheet, header=None, nrows=20, dtype=str)

    def row_has(index: int, pattern: str) -> bool:
        return metadata.iloc[index].astype(str).str.fullmatch(pattern, case=False).any()

    date_row = next((index for index in metadata.index if row_has(index, r"Date")), None)
    time_row = next((index for index in metadata.index if row_has(index, r"Time")), None)
    require(date_row is not None and time_row is not None, f"Missing 'Date'/'Time' rows in sheet {sheet!r}")
    date = pd.to_datetime(metadata.iloc[date_row, 1]).date()
    time = pd.to_datetime(metadata.iloc[time_row, 1]).time()
    return datetime.combine(date, time)


def normalize_time_series(
    time: pd.Series,
    *,
    time_round_decimals: int | None,
    time_step_h: float | None,
) -> pd.Series:
    normalized = pd.to_numeric(time, errors="raise")
    if time_step_h is not None:
        step = float(time_step_h)
        require(step > 0, "time_step_h must be > 0")
        normalized = (normalized / step).round() * step
    if time_round_decimals is not None:
        decimals = int(time_round_decimals)
        require(decimals >= 0, "time_round_decimals must be >= 0")
        normalized = normalized.round(decimals)
    return normalized


def canonical_channel(value: str) -> str:
    raw = str(value or "").strip()
    prefix, separator, suffix = raw.partition(":")
    normalized_prefix = re.sub(r"\s+[AB]$", "", prefix.strip(), flags=re.IGNORECASE)
    normalized_prefix = re.sub(r"\s+", " ", normalized_prefix).strip()
    if not separator:
        return normalized_prefix
    normalized_suffix = re.sub(r"\s+", "", suffix)
    return f"{normalized_prefix}:{normalized_suffix}"


_OVERFLOW_TOKENS = frozenset({"overflow", "ovrflw", "ovr", "over", "inf", "infinity", "∞"})
_GREATER_THAN_NUMBER = re.compile(r"^>\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def is_overflow_token(value: object) -> bool:
    normalized = str(value or "").strip()
    if not normalized:
        return False
    return normalized.lower() in _OVERFLOW_TOKENS or bool(_GREATER_THAN_NUMBER.fullmatch(normalized))


def normalize_channel_map(channel_map: Mapping[str, str] | None) -> Mapping[str, str]:
    if not channel_map:
        return {}
    normalized: dict[str, str] = {}
    for raw_label, output_channel in channel_map.items():
        key = canonical_channel(str(raw_label)).lower()
        value = str(output_channel).strip()
        require(key != "", "channel_map keys must contain a channel label")
        require(value != "", f"channel_map[{raw_label!r}] must name an output channel")
        require(
            key not in normalized,
            f"Duplicate channel_map declaration after normalization: {raw_label!r} resolves to {key!r}",
        )
        normalized[key] = value
    return normalized


def resolve_channel_from_map(raw_label: str, *, channel_map_ci: Mapping[str, str]) -> str | None:
    label = canonical_channel(raw_label).lower()
    return channel_map_ci.get(label)


def resolve_channel(
    raw_label: str,
    *,
    channels: Sequence[str] | None,
    channel_map_ci: Mapping[str, str],
) -> str | None:
    normalized_label = canonical_channel(raw_label).lower()
    mapped = resolve_channel_from_map(raw_label, channel_map_ci=channel_map_ci)
    if mapped is not None:
        return mapped

    if channels:
        declared = [(canonical_channel(channel).lower(), channel) for channel in channels]
        matches = [channel for normalized, channel in declared if normalized == normalized_label]
        if not matches and not channel_map_ci and ":" in normalized_label:
            base_label = normalized_label.partition(":")[0]
            matches = [channel for normalized, channel in declared if normalized == base_label]
        require(
            len(matches) <= 1,
            f"Ambiguous configured channels for raw label {raw_label!r}: {matches}",
        )
        return matches[0] if matches else None

    if channel_map_ci:
        return None
    raise ValueError("Provide at least one of: channels or channel_map")


def time_column(header: Iterable[str]) -> str:
    for value in header:
        if str(value).strip().lower().startswith("time"):
            return str(value)
    raise ValueError("Time column not found in kinetic block header")


def well_headers(header: Iterable[str]) -> list[str]:
    candidates = [str(value).strip() for value in header if re.fullmatch(r"[A-H][0-9]+", str(value).strip())]
    invalid = [value for value in candidates if not re.fullmatch(r"[A-H](?:[1-9]|1[0-2])", value)]
    require(not invalid, f"Kinetic header contains positions outside the 96-well range A1..H12: {invalid}")
    duplicates = sorted({value for value in candidates if candidates.count(value) > 1})
    require(not duplicates, f"Kinetic header contains duplicate well columns: {duplicates}")
    return candidates


def is_blank_measurement(value: object) -> bool:
    return value is None or bool(pd.isna(value)) or str(value).strip() == ""


def coerce_measurements(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    result = frame.loc[~frame["value"].map(is_blank_measurement)].copy()
    if result.empty:
        result["overflow"] = pd.Series(dtype=bool)
        result["value"] = pd.Series(dtype=float)
        return result

    raw_values = result["value"].astype(str)
    overflow = raw_values.map(is_overflow_token)
    numeric = pd.to_numeric(raw_values, errors="coerce")
    invalid = numeric.isna() & ~overflow
    if invalid.any():
        index = invalid[invalid].index[0]
        row = result.loc[index]
        raise ValueError(
            f"Invalid {source} measurement token {raw_values.loc[index]!r} in sheet {row['sheet_name']!r}, "
            f"channel {row['channel']!r}, well {row['position']!r}"
        )

    result["overflow"] = overflow.astype(bool)
    result["value"] = numeric.astype(float)
    result.loc[overflow, "value"] = np.inf
    return result.reset_index(drop=True)
