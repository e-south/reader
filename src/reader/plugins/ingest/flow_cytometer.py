"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/ingest/flow_cytometer.py

Flow cytometer ingest for .fcs files (snapshot data).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from reader.core.errors import ParseError
from reader.core.registry import Plugin, PluginConfig
from reader.io.discovery import DEFAULT_EXCLUDE, DEFAULT_ROOTS, discover_files

DEFAULT_FCS_INCLUDE = ("*.fcs", "*.FCS")


class FlowCytometerCfg(PluginConfig):
    # auto-discovery knobs
    auto_roots: list[str] | None = None
    auto_include: list[str] = Field(default_factory=lambda: list(DEFAULT_FCS_INCLUDE))
    auto_exclude: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    auto_pick: Literal["single", "latest", "merge"] = "merge"
    auto_recursive: bool = False

    # channel naming / output shaping
    channel_name_field: str = "pns"  # field in FCS channel metadata (e.g., pns or pnn)
    channel_map: Mapping[str, str] | None = None
    drop_channels: list[str] | None = None
    sample_id_from: Literal["stem", "name"] = "stem"
    time_value: float = 0.0

    # logging
    print_summary: bool = True


class FlowCytometerIngest(Plugin):
    """Ingest .fcs files into tidy.v1 (snapshot; time is constant)."""

    key = "flow_cytometer"
    category = "ingest"
    ConfigModel = FlowCytometerCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"raw?": "none"}  # optional explicit file input

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1", "channels": "cytometer.channels.v1"}

    def _auto_pick_one(self, files: list[Path], mode: str) -> Path:
        if mode == "single":
            if len(files) != 1:
                raise ParseError(
                    "Auto-discovery expected exactly one .fcs file, found "
                    f"{len(files)}:\n- "
                    + "\n- ".join(str(p) for p in files)
                    + "\nHint: set auto_pick: latest or auto_pick: merge, or pass reads.raw explicitly."
                )
            return files[0]
        if mode == "latest":
            return max(files, key=lambda p: p.stat().st_mtime)
        raise ParseError(f"_auto_pick_one called with mode={mode!r}")

    def _discover(self, ctx, cfg: FlowCytometerCfg) -> list[Path]:
        roots = cfg.auto_roots or list(DEFAULT_ROOTS)
        files = discover_files(
            ctx.exp_dir,
            roots=roots,
            include=cfg.auto_include,
            exclude=cfg.auto_exclude,
            recursive=cfg.auto_recursive,
        )
        if not files:
            raise ParseError(
                f"No .fcs files discovered under {roots} (include={cfg.auto_include}, exclude={cfg.auto_exclude}).\n"
                "Hint: put raw files under ./inputs (default), or set auto_roots / reads.raw explicitly."
            )
        if cfg.auto_pick in ("single", "latest"):
            return [self._auto_pick_one(files, cfg.auto_pick)]
        if cfg.auto_pick == "merge":
            return files
        raise ParseError(f"Unknown auto_pick mode {cfg.auto_pick!r} (expected: single|latest|merge)")

    def _channel_names(self, channels: dict[int, dict[str, object]], *, field: str) -> list[str]:
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

    def run(self, ctx, inputs, cfg: FlowCytometerCfg):
        try:
            from flowio import FlowData  # noqa: PLC0415
        except Exception as e:  # pragma: no cover - environment-specific
            raise ParseError(
                "flowio is required for ingest/flow_cytometer. Install with: uv sync --locked --group cytometry"
            ) from e

        files = [inputs["raw"]] if "raw" in inputs else self._discover(ctx, cfg)
        field = str(cfg.channel_name_field).lower().strip()
        channel_map = {str(k): str(v) for k, v in (cfg.channel_map or {}).items()}
        drop_channels = {str(c) for c in (cfg.drop_channels or [])}

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

        frames: list[pd.DataFrame] = []
        channel_meta_frames: list[pd.DataFrame] = []
        for f in files:
            flow = FlowData(str(f))
            event_count = int(flow.event_count)
            channel_count = int(flow.channel_count)
            raw_events = np.asarray(flow.events, dtype=float)
            if raw_events.size != event_count * channel_count:
                raise ParseError(
                    f"Unexpected event buffer size for {f.name}: "
                    f"{raw_events.size} values for {event_count} events × {channel_count} channels."
                )
            values = raw_events.reshape(event_count, channel_count)
            channel_names = self._channel_names(flow.channels, field=field)
            if len(channel_names) != channel_count:
                raise ParseError(
                    f"Channel count mismatch: metadata has {len(channel_names)} names, events have {channel_count}."
                )
            if channel_map:
                mapped = [channel_map.get(name, name) for name in channel_names]
                if len(set(mapped)) != len(mapped):
                    raise ParseError("channel_map produces duplicate channel names; ensure a 1:1 mapping.")
                channel_names = mapped
            wide = pd.DataFrame(values, columns=channel_names)
            wide["event_index"] = range(event_count)
            long = wide.melt(id_vars=["event_index"], var_name="channel", value_name="value")
            if drop_channels:
                long = long[~long["channel"].isin(drop_channels)]
            sample_id = f.stem if cfg.sample_id_from == "stem" else f.name
            long["sample_id"] = sample_id
            long["position"] = sample_id
            long["time"] = float(cfg.time_value)
            frames.append(long)

            channel_rows = []
            channel_indices = sorted(flow.channels)
            for idx, name in zip(channel_indices, channel_names, strict=False):
                meta = flow.channels.get(idx, {})
                pne_decades, pne_zero = _parse_pne(meta.get("pne"))
                row = {
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
                channel_rows.append(row)
            if channel_rows:
                channel_meta_frames.append(pd.DataFrame(channel_rows))

        if not frames:
            raise ParseError("No cytometer frames parsed from selected files")
        out = pd.concat(frames, ignore_index=True)

        if cfg.print_summary:
            with suppress(Exception):
                ctx.logger.info(
                    "flow_cytometer ingest • files=%d • rows=%d • channels=%d • samples=%d",
                    len(files),
                    len(out),
                    out["channel"].nunique(),
                    out["sample_id"].nunique(),
                )

        if channel_meta_frames:
            channels_meta = pd.concat(channel_meta_frames, ignore_index=True)
        else:
            channels_meta = pd.DataFrame(columns=["sample_id", "channel_index", "channel_name"])

        return {"df": out, "channels": channels_meta}
