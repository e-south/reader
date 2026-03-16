"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/cytometry/io/fcs.py

FCS parsing for cytometry experiments.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from reader.errors import ParseError


def _to_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _parse_pne(value) -> tuple[float, float]:
    if value is None:
        return (float("nan"), float("nan"))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (_to_float(value[0]), _to_float(value[1]))
    text = str(value).strip()
    if "," in text:
        left, right = text.split(",", 1)
        return (_to_float(left.strip()), _to_float(right.strip()))
    return (float("nan"), float("nan"))


def _clean_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return value.decode(errors="ignore")
    return str(value)


def _channel_names(channels: dict[int, dict[str, object]], *, field: str) -> list[str]:
    names: list[str] = []
    for key in sorted(channels):
        meta = channels[key]
        name = meta.get(field)
        if name is None:
            raise ParseError(
                f"Channel metadata missing field '{field}' for channel {key}. Use channel_name_field: pns or pnn."
            )
        names.append(str(name))
    return names


def parse_fcs_file(
    path: Path,
    *,
    channel_name_field: str,
    channel_map: Mapping[str, str] | None = None,
    drop_channels: set[str] | None = None,
    sample_id_from: Literal["stem", "name"] = "stem",
    time_value: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from flowio import FlowData  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment-specific
        raise ParseError(
            "flowio is required for ingest/flow_cytometer. Install with: uv sync --locked --group cytometry"
        ) from exc

    field = str(channel_name_field).lower().strip()
    mapped_names = {str(k): str(v) for k, v in (channel_map or {}).items()}
    dropped = {str(channel) for channel in (drop_channels or set())}

    flow = FlowData(str(path))
    event_count = int(flow.event_count)
    channel_count = int(flow.channel_count)
    raw_events = np.asarray(flow.events, dtype=float)
    if raw_events.size != event_count * channel_count:
        raise ParseError(
            f"Unexpected event buffer size for {path.name}: "
            f"{raw_events.size} values for {event_count} events × {channel_count} channels."
        )

    values = raw_events.reshape(event_count, channel_count)
    channel_names = _channel_names(flow.channels, field=field)
    if len(channel_names) != channel_count:
        raise ParseError(
            f"Channel count mismatch: metadata has {len(channel_names)} names, events have {channel_count}."
        )
    if mapped_names:
        remapped = [mapped_names.get(name, name) for name in channel_names]
        if len(set(remapped)) != len(remapped):
            raise ParseError("channel_map produces duplicate channel names; ensure a 1:1 mapping.")
        channel_names = remapped

    wide = pd.DataFrame(values, columns=channel_names)
    wide["event_index"] = range(event_count)
    long = wide.melt(id_vars=["event_index"], var_name="channel", value_name="value")
    if dropped:
        long = long[~long["channel"].isin(dropped)]
    sample_id = path.stem if sample_id_from == "stem" else path.name
    long["sample_id"] = sample_id
    long["position"] = sample_id
    long["time"] = float(time_value)

    channel_rows = []
    channel_indices = sorted(flow.channels)
    for idx, name in zip(channel_indices, channel_names, strict=False):
        meta = flow.channels.get(idx, {})
        pne_decades, pne_zero = _parse_pne(meta.get("pne"))
        channel_rows.append(
            {
                "sample_id": sample_id,
                "channel_index": int(idx),
                "channel_name": str(name),
                "pns": _clean_text(meta.get("pns")),
                "pnn": _clean_text(meta.get("pnn")),
                "pnt": _clean_text(meta.get("pnt")),
                "pnf": _clean_text(meta.get("pnf")),
                "pnl": _clean_text(meta.get("pnl")),
                "pnr": _to_float(meta.get("pnr")),
                "pnb": _to_float(meta.get("pnb")),
                "png": _to_float(meta.get("png")),
                "pne_decades": pne_decades,
                "pne_zero": pne_zero,
            }
        )
    channels_meta = pd.DataFrame(channel_rows)
    if channels_meta.empty:
        channels_meta = pd.DataFrame(columns=["sample_id", "channel_index", "channel_name"])
    return long, channels_meta
