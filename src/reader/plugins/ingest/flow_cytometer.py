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
        return {"df": "tidy.v1"}

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
                    f"Channel metadata missing field '{field}' for channel {key}. "
                    "Use channel_name_field: pns or pnn."
                )
            names.append(str(name))
        return names

    def run(self, ctx, inputs, cfg: FlowCytometerCfg):
        try:
            from flowio import FlowData
        except Exception as e:  # pragma: no cover - environment-specific
            raise ParseError(
                "flowio is required for ingest/flow_cytometer. "
                "Install with: uv sync --locked --group cytometry"
            ) from e

        files = [inputs["raw"]] if "raw" in inputs else self._discover(ctx, cfg)
        field = str(cfg.channel_name_field).lower().strip()
        channel_map = {str(k): str(v) for k, v in (cfg.channel_map or {}).items()}
        drop_channels = {str(c) for c in (cfg.drop_channels or [])}

        frames: list[pd.DataFrame] = []
        for f in files:
            flow = FlowData(str(f))
            event_count = int(flow.event_count)
            channel_count = int(flow.channel_count)
            values = np.asarray(flow.events, dtype=float).reshape(event_count, channel_count)
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

        return {"df": out}
