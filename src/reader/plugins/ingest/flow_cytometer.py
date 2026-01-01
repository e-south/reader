"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/ingest/flow_cytometer.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from reader.core.errors import ParseError
from reader.core.registry import Plugin, PluginConfig
from reader.parsers.discovery import DEFAULT_EXCLUDE, DEFAULT_ROOTS, discover_files
from reader.parsers.flow_cytometer import parse_fcs_events

DEFAULT_FCS_INCLUDE = ("*.fcs",)


class FlowCytometerCfg(PluginConfig):
    # auto-discovery knobs
    auto_roots: list[str] | None = None
    auto_include: list[str] = list(DEFAULT_FCS_INCLUDE)
    auto_exclude: list[str] = list(DEFAULT_EXCLUDE)
    auto_pick: Literal["single", "latest", "merge"] = "merge"
    auto_recursive: bool = False

    # parsing knobs
    channels: list[str] | None = None
    channel_map: Mapping[str, str] | None = None
    channel_name_field: Literal["pns", "pnn"] = "pns"
    include_source_file: bool = True

    # logging
    print_summary: bool = True


class FlowCytometer(Plugin):
    """FCS ingest (event-level tidy table)."""

    key = "flow_cytometer"
    category = "ingest"
    ConfigModel = FlowCytometerCfg

    @classmethod
    def input_contracts(cls):
        return {"raw?": "none"}  # file(s) optional; can auto-discover

    @classmethod
    def output_contracts(cls):
        return {"df": "cyto.events.v1"}

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
                f"No .fcs discovered under {roots} (include={cfg.auto_include}, exclude={cfg.auto_exclude})."
            )
        if cfg.auto_pick in ("single", "latest"):
            return [self._auto_pick_one(files, cfg.auto_pick)]
        if cfg.auto_pick == "merge":
            return files
        raise ParseError(f"Unknown auto_pick mode {cfg.auto_pick!r} (expected: single|latest|merge)")

    def _log_df_summary(self, ctx, df: pd.DataFrame, files_count: int):
        try:
            n_rows = len(df)
            n_events = df["event_index"].nunique()
            n_samples = df["sample_id"].nunique()
            chans = sorted(df["channel"].astype(str).unique().tolist())
            ctx.logger.info(
                "Flow cytometer ingest • files=%d • rows=%d • events=%d • samples=%d • channels=%d",
                files_count,
                n_rows,
                n_events,
                n_samples,
                len(chans),
            )
            if chans:
                preview = ", ".join(chans[:8]) + (" …" if len(chans) > 8 else "")
                ctx.logger.info("channels: %s", preview)
        except Exception:
            pass

    def run(self, ctx, inputs, cfg: FlowCytometerCfg):
        try:
            if "raw" in inputs:
                raw = inputs["raw"]
                files = [Path(p) for p in (raw if isinstance(raw, Sequence) and not isinstance(raw, (str, Path)) else [raw])]
            else:
                files = self._discover(ctx, cfg)

            frames: list[pd.DataFrame] = []
            for f in files:
                df = parse_fcs_events(
                    f,
                    channels=cfg.channels,
                    channel_map=cfg.channel_map,
                    channel_name_field=cfg.channel_name_field,
                    include_source_file=cfg.include_source_file,
                )
                frames.append(df)

            if not frames:
                raise ParseError("No frames parsed from selected files")
            out = pd.concat(frames, ignore_index=True)

            if cfg.print_summary:
                self._log_df_summary(ctx, out, files_count=len(files))

        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Flow cytometer ingest failed: {e}") from e

        return {"df": out}
