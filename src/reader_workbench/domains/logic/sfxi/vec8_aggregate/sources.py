from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd

from reader_workbench.errors import SFXIError

from .checks import finite_numeric_column, require_vec8_columns
from .constants import METADATA_COLUMNS, VEC8_CHANNELS
from .model import SFXIVec8Aggregate, SFXIVec8Source


def aggregate_sfxi_vec8_sources(
    sources: list[SFXIVec8Source] | tuple[SFXIVec8Source, ...],
) -> SFXIVec8Aggregate:
    """Validate and combine already-resolved vec8 sources.

    Resolving experiment configurations and record catalogs is a runtime concern.
    This domain operation accepts explicit source data and owns only SFXI vec8
    validation and normalization.
    """
    if not sources:
        raise SFXIError("SFXI vec8 aggregate requires at least one source.")
    _require_unique_source_records(sources)

    frames: list[pd.DataFrame] = []
    for source_index, source in enumerate(sources):
        normalized = _normalize_vec8_frame(source, source_index=source_index)
        frames.append(normalized)

    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise SFXIError("SFXI vec8 aggregate has no rows to plot.")
    return SFXIVec8Aggregate(frame=frame)


def _normalize_vec8_frame(source: SFXIVec8Source, *, source_index: int) -> pd.DataFrame:
    resource_id = _nonempty_identity(source.resource_id, field="resource_id")
    experiment_id = _nonempty_identity(source.experiment_id, field="experiment_id")
    record_id = _nonempty_identity(source.record_id, field="record_id")
    revision_digest = _canonical_sha256_digest(source.revision_digest)
    source_label = f"{resource_id} ({experiment_id}:{record_id})"
    frame = source.frame.copy()
    require_vec8_columns(frame)
    if frame.empty:
        raise SFXIError(f"SFXI vec8 source has no rows: {source_label}")

    out = frame.reset_index(drop=True)
    for channel in VEC8_CHANNELS:
        out[channel] = finite_numeric_column(out[channel], column=channel, source=source_label)
    out["design_id"] = _nonempty_string_column(out["design_id"], column="design_id", source=source_label)
    if "time_selected_h" in out.columns:
        out["time_selected_h"] = finite_numeric_column(
            out["time_selected_h"],
            column="time_selected_h",
            source=source_label,
            allow_nan=True,
        )
    out["reference_design_id"] = _nonempty_string_column(
        out["reference_design_id"], column="reference_design_id", source=source_label
    )
    out["intensity_log2_offset_delta"] = _nonnegative_numeric_column(
        out["intensity_log2_offset_delta"], column="intensity_log2_offset_delta", source=source_label
    )
    out["r_logic"] = _nonnegative_numeric_column(out["r_logic"], column="r_logic", source=source_label)
    out["flat_logic"] = _strict_bool_column(out["flat_logic"], column="flat_logic", source=source_label)
    if out["design_id"].duplicated().any():
        duplicates = sorted(out.loc[out["design_id"].duplicated(keep=False), "design_id"].unique())
        raise SFXIError(
            "SFXI vec8 aggregate design_id values must be unique within each source: " + ", ".join(duplicates)
        )
    out.insert(0, "source_row_index", range(len(out)))
    out.insert(0, "source_record_revision_digest", revision_digest)
    out.insert(0, "source_record_id", record_id)
    out.insert(0, "source_experiment_id", experiment_id)
    out.insert(0, "source_resource_id", resource_id)
    out.insert(0, "source_index", int(source_index))
    out["row_label"] = _row_labels(out)

    ordered = [
        *METADATA_COLUMNS,
        *[column for column in VEC8_CHANNELS if column in out.columns],
    ]
    ordered = [column for column in ordered if column in out.columns]
    ordered += [column for column in out.columns if column not in set(ordered)]
    return out.loc[:, ordered]


def _require_unique_source_records(sources: list[SFXIVec8Source] | tuple[SFXIVec8Source, ...]) -> None:
    identities = [
        (
            _nonempty_identity(source.experiment_id, field="experiment_id"),
            _nonempty_identity(source.record_id, field="record_id"),
        )
        for source in sources
    ]
    duplicates = sorted(identity for identity, count in Counter(identities).items() if count > 1)
    if duplicates:
        formatted = ", ".join(f"{experiment_id}:{record_id}" for experiment_id, record_id in duplicates)
        raise SFXIError(f"SFXI vec8 aggregate source record identities must be unique: {formatted}")


def _nonempty_identity(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SFXIError(f"SFXI vec8 aggregate {field} must be a non-empty string.")
    return value.strip()


def _canonical_sha256_digest(value: str) -> str:
    if isinstance(value, str) and value.startswith("sha256:"):
        digest = value.removeprefix("sha256:")
        if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
            return value
    raise SFXIError("SFXI vec8 aggregate revision_digest must be a canonical sha256 digest.")


def _nonempty_string_column(series: pd.Series, *, column: str, source: str) -> pd.Series:
    values = series.astype("string")
    invalid = values.isna() | values.str.strip().eq("")
    if invalid.any():
        raise SFXIError(f"SFXI vec8 aggregate column {column!r} must contain non-empty labels in {source}.")
    return values.astype(str)


def _nonnegative_numeric_column(series: pd.Series, *, column: str, source: str) -> pd.Series:
    values = finite_numeric_column(series, column=column, source=source)
    if (values < 0.0).any():
        raise SFXIError(f"SFXI vec8 aggregate column {column!r} must contain nonnegative values in {source}.")
    return values


def _strict_bool_column(series: pd.Series, *, column: str, source: str) -> pd.Series:
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
    labels = frame["source_resource_id"].astype(str) + " :: " + frame["design_id"].astype(str)
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
