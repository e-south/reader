"""
--------------------------------------------------------------------------------
<reader project>
src/reader/parsers/flow_cytometer.py

Parse FCS files into a tidy, event-level table.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from flowio import FlowData
except Exception:  # optional dependency; raised when parse is invoked
    FlowData = None


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _channel_names(fd: Any, field: Literal["pns", "pnn"]) -> list[str]:
    names: list[str] = []
    for idx in sorted(fd.channels):
        meta = fd.channels[idx]
        name = meta.get(field) or meta.get("pnn") or meta.get("pns") or f"CH{idx}"
        names.append(str(name))
    return names


def _apply_channel_map(names: list[str], channel_map: Mapping[str, str] | None) -> list[str]:
    if not channel_map:
        return names
    lower = {str(k).lower(): str(v) for k, v in channel_map.items()}
    out: list[str] = []
    for n in names:
        out.append(channel_map.get(n, lower.get(n.lower(), n)))
    return out


def parse_fcs_events(
    path: str | Path,
    *,
    channel_map: Mapping[str, str] | None = None,
    channels: Sequence[str] | None = None,
    channel_name_field: Literal["pns", "pnn"] = "pns",
    sample_id: str | None = None,
    include_source_file: bool = True,
) -> pd.DataFrame:
    """
    Parse an FCS file into a tidy event table with columns:
      sample_id, event_index, channel, value (+ optional source_file)
    """
    p = Path(path)
    _require(p.exists(), f"Input file not found: {p}")
    _require(p.suffix.lower() == ".fcs", f"Unsupported file type: {p.suffix}")

    if FlowData is None:
        raise ValueError("flowio is required to parse FCS files; install with `uv sync --extra cytometry`.")

    fd = FlowData(str(p))
    par = int(fd.text.get("par", 0))
    _require(par > 0, f"Invalid FCS header: par={par}")

    events = np.asarray(fd.events, dtype=float)
    _require(events.size % par == 0, f"FCS event buffer size not divisible by par ({events.size} % {par})")
    n_events = events.size // par
    arr = events.reshape((n_events, par))

    names = _channel_names(fd, channel_name_field)
    _require(len(names) == par, "Channel count does not match par")

    names = _apply_channel_map(names, channel_map)
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate channel names after mapping: {names}")

    if channels is not None:
        keep = set(channels)
        missing = sorted(set(channels) - set(names))
        _require(not missing, f"Requested channels not found: {missing}")
        idx = [i for i, n in enumerate(names) if n in keep]
        names = [names[i] for i in idx]
        arr = arr[:, idx]

    df = pd.DataFrame(arr, columns=names)
    sid = sample_id or p.stem
    df.insert(0, "event_index", np.arange(n_events, dtype="int64"))
    df.insert(0, "label_id", sid)
    df.insert(0, "sample_id", sid)
    if include_source_file:
        df.insert(3, "source_file", p.name)

    id_vars = ["sample_id", "label_id", "event_index"] + (["source_file"] if include_source_file else [])
    tidy = df.melt(id_vars=id_vars, var_name="channel", value_name="value")
    tidy["event_index"] = tidy["event_index"].astype("int64")
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    if tidy["value"].isna().any():
        raise ValueError("Parsed FCS contains non-numeric values")

    return tidy
