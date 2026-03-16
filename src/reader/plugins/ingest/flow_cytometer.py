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

import pandas as pd
from pydantic import Field

from reader.core.errors import ParseError
from reader.domains.cytometry.io.fcs import parse_fcs_file
from reader.plugins.ingest._discovery import discover_auto_input_files
from reader.plugins.ingest.discovery_policy import DEFAULT_EXCLUDE
from reader.workbench.ports import dataframe_output, file_path_input
from reader.workbench.registry import Plugin, PluginConfig

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

    ConfigModel = FlowCytometerCfg

    @classmethod
    def input_ports(cls):
        return {"raw": file_path_input("raw", optional=True)}

    @classmethod
    def output_ports(cls):
        return {
            "df": dataframe_output("df", "tidy.v1"),
            "channels": dataframe_output("channels", "cytometer.channels.v1"),
        }

    def _discover(self, ctx, cfg: FlowCytometerCfg) -> list[Path]:
        return discover_auto_input_files(
            exp_dir=ctx.exp_dir,
            auto_roots=cfg.auto_roots,
            auto_include=cfg.auto_include,
            auto_exclude=cfg.auto_exclude,
            auto_recursive=cfg.auto_recursive,
            auto_pick=cfg.auto_pick,
            discovery_label=".fcs files",
            singular_label=".fcs file",
        )

    def run(self, ctx, inputs, cfg: FlowCytometerCfg):
        files = [inputs["raw"]] if "raw" in inputs else self._discover(ctx, cfg)
        channel_map = {str(k): str(v) for k, v in (cfg.channel_map or {}).items()}
        drop_channels = {str(c) for c in (cfg.drop_channels or [])}

        frames: list[pd.DataFrame] = []
        channel_meta_frames: list[pd.DataFrame] = []
        for f in files:
            long, channels_meta = parse_fcs_file(
                f,
                channel_name_field=cfg.channel_name_field,
                channel_map=channel_map,
                drop_channels=drop_channels,
                sample_id_from=cfg.sample_id_from,
                time_value=cfg.time_value,
            )
            frames.append(long)
            if not channels_meta.empty:
                channel_meta_frames.append(channels_meta)

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
