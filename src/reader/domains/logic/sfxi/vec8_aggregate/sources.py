from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from reader.errors import SFXIError

from .checks import finite_numeric_column, require_vec8_columns
from .constants import DIRECT_TABLE_SUFFIXES, METADATA_COLUMNS, VEC8_CHANNELS
from .model import LoadedSFXIVec8Source, SFXIVec8Aggregate, SFXIVec8Source


def aggregate_sfxi_vec8_sources(
    sources: list[LoadedSFXIVec8Source] | tuple[LoadedSFXIVec8Source, ...],
) -> SFXIVec8Aggregate:
    """Validate and combine already-resolved vec8 sources.

    Resolving experiment configurations and record catalogs is a runtime concern.
    This domain operation accepts explicit source data and owns only SFXI vec8
    validation and normalization.
    """
    if not sources:
        raise SFXIError("SFXI vec8 aggregate requires at least one source.")

    frames: list[pd.DataFrame] = []
    source_records: list[SFXIVec8Source] = []
    for source_index, loaded in enumerate(sources):
        normalized = _normalize_vec8_frame(loaded, source_index=source_index)
        frames.append(normalized)
        source_records.append(
            SFXIVec8Source(
                source_id=loaded.source_id,
                source_path=loaded.source_path,
                table_path=loaded.table_path,
                source_kind=loaded.source_kind,
                row_count=len(normalized),
                record_id=loaded.record_id,
                record_metadata=loaded.record_metadata,
            )
        )

    _require_unique_source_ids(source_records)
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise SFXIError("SFXI vec8 aggregate has no rows to plot.")
    return SFXIVec8Aggregate(frame=frame, sources=tuple(source_records))


def load_sfxi_vec8_table(path: str | Path) -> LoadedSFXIVec8Source:
    """Load one explicit vec8 table without consulting workbench state."""

    path = Path(path).expanduser()
    resolved = path.resolve()
    if not resolved.exists():
        raise SFXIError(f"SFXI vec8 aggregate source does not exist: {resolved}")
    if not resolved.is_file():
        raise SFXIError(f"SFXI vec8 aggregate table source must be a file: {resolved}")
    if resolved.suffix.lower() in DIRECT_TABLE_SUFFIXES:
        return _load_direct_table(resolved)
    raise SFXIError(f"Unsupported SFXI vec8 table format: {resolved}")


def _load_direct_table(path: Path) -> LoadedSFXIVec8Source:
    return LoadedSFXIVec8Source(
        source_id=_direct_table_source_id(path),
        source_path=path.resolve(),
        table_path=path.resolve(),
        source_kind="table",
        frame=_read_table(path),
    )


def _direct_table_source_id(path: Path) -> str:
    for parent in path.resolve().parents:
        if (parent / "config.yaml").exists():
            return parent.name
    return path.stem


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, sheet_name="vec8")
    except ValueError as exc:
        if suffix in {".xlsx", ".xls"} and "Worksheet named 'vec8' not found" in str(exc):
            raise SFXIError(f"SFXI vec8 workbook must include a 'vec8' sheet: {path}") from exc
        raise SFXIError(f"SFXI vec8 aggregate could not read table {path}: {exc}") from exc
    except Exception as exc:
        raise SFXIError(f"SFXI vec8 aggregate could not read table {path}: {exc}") from exc
    raise SFXIError(f"Unsupported SFXI vec8 table format: {path}")


def _normalize_vec8_frame(loaded: LoadedSFXIVec8Source, *, source_index: int) -> pd.DataFrame:
    frame = loaded.frame.copy()
    require_vec8_columns(frame)
    if frame.empty:
        raise SFXIError(f"SFXI vec8 source has no rows: {loaded.source_path}")

    out = frame.reset_index(drop=True)
    for channel in VEC8_CHANNELS:
        out[channel] = finite_numeric_column(out[channel], column=channel, source=loaded.source_path)
    out["design_id"] = _nonempty_string_column(out["design_id"], column="design_id", source=loaded.source_path)
    if "time_selected_h" in out.columns:
        out["time_selected_h"] = finite_numeric_column(
            out["time_selected_h"],
            column="time_selected_h",
            source=loaded.source_path,
            allow_nan=True,
        )
    out["reference_design_id"] = _nonempty_string_column(
        out["reference_design_id"], column="reference_design_id", source=loaded.source_path
    )
    out["intensity_log2_offset_delta"] = _nonnegative_numeric_column(
        out["intensity_log2_offset_delta"], column="intensity_log2_offset_delta", source=loaded.source_path
    )
    out["r_logic"] = _nonnegative_numeric_column(out["r_logic"], column="r_logic", source=loaded.source_path)
    out["flat_logic"] = _strict_bool_column(out["flat_logic"], column="flat_logic", source=loaded.source_path)
    if out["design_id"].duplicated().any():
        duplicates = sorted(out.loc[out["design_id"].duplicated(keep=False), "design_id"].unique())
        raise SFXIError(
            "SFXI vec8 aggregate design_id values must be unique within each source: " + ", ".join(duplicates)
        )
    out.insert(0, "source_row_index", range(len(out)))
    out.insert(0, "source_kind", loaded.source_kind)
    out.insert(0, "table_path", str(loaded.table_path))
    out.insert(0, "source_path", str(loaded.source_path))
    out.insert(0, "source_id", loaded.source_id)
    out.insert(0, "source_index", int(source_index))
    out["row_label"] = _row_labels(out)

    ordered = [
        *METADATA_COLUMNS,
        *[column for column in VEC8_CHANNELS if column in out.columns],
    ]
    ordered = [column for column in ordered if column in out.columns]
    ordered += [column for column in out.columns if column not in set(ordered)]
    return out.loc[:, ordered]


def _require_unique_source_ids(sources: list[SFXIVec8Source]) -> None:
    values = pd.Series([source.source_id for source in sources], dtype="string")
    duplicates = sorted(values[values.duplicated(keep=False)].dropna().unique().tolist())
    if duplicates:
        raise SFXIError("SFXI vec8 aggregate source_id values must be unique: " + ", ".join(map(str, duplicates)))


def _nonempty_string_column(series: pd.Series, *, column: str, source: Path) -> pd.Series:
    values = series.astype("string")
    invalid = values.isna() | values.str.strip().eq("")
    if invalid.any():
        raise SFXIError(f"SFXI vec8 aggregate column {column!r} must contain non-empty labels in {source}.")
    return values.astype(str)


def _nonnegative_numeric_column(series: pd.Series, *, column: str, source: Path) -> pd.Series:
    values = finite_numeric_column(series, column=column, source=source)
    if (values < 0.0).any():
        raise SFXIError(f"SFXI vec8 aggregate column {column!r} must contain nonnegative values in {source}.")
    return values


def _strict_bool_column(series: pd.Series, *, column: str, source: Path) -> pd.Series:
    parsed: list[bool] = []
    invalid = False
    for value in series.tolist():
        if isinstance(value, bool):
            parsed.append(value)
            continue
        if pd.isna(value):
            invalid = True
            break
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                parsed.append(True)
                continue
            if normalized == "false":
                parsed.append(False)
                continue
        invalid = True
        break
    if invalid:
        raise SFXIError(f"SFXI vec8 aggregate column {column!r} must contain boolean values in {source}.")
    return pd.Series(parsed, index=series.index, dtype=bool)


def _row_labels(frame: pd.DataFrame) -> pd.Series:
    labels = frame["source_id"].astype(str) + " :: " + frame["design_id"].astype(str)
    if not labels.duplicated().any():
        return labels
    if "time_selected_h" in frame.columns:
        labels = labels + " @ " + frame["time_selected_h"].map(_format_time_label)
    if not labels.duplicated().any():
        return labels
    duplicate_index = labels.groupby(labels).cumcount() + 1
    return labels + " #" + duplicate_index.astype(str)


def _format_time_label(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:g}h"
