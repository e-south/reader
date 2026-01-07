"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/ingest/synergy_h1.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd

from reader.core.errors import ParseError
from reader.core.registry import Plugin, PluginConfig
from reader.io.discovery import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    DEFAULT_ROOTS,
    discover_files,
)
from reader.io.synergy_h1 import parse_kinetic_only, parse_snapshot_and_timeseries


class SynergyH1UnifiedCfg(PluginConfig):
    # What kind of ingest are we doing?
    mode: Literal["auto", "snapshot_only", "kinetic_only", "mixed"] = "auto"

    # parsing knobs
    channels: list[str] | None = None
    channel_map: Mapping[str, str] | None = None
    sheet_names: Sequence[str] | None = None
    add_sheet: bool = False

    # auto-discovery knobs
    auto_roots: list[str] | None = None
    auto_include: list[str] = list(DEFAULT_INCLUDE)
    auto_exclude: list[str] = list(DEFAULT_EXCLUDE)
    auto_pick: str = "single"  # single | latest | merge
    auto_recursive: bool = False

    # Optionally annotate source on merge mode
    add_source_column: bool = False
    source_col: str = "source_file"

    # logging
    print_summary: bool = True


class SynergyH1(Plugin):
    """Unified Synergy H1 ingest (snapshot-only, kinetic-only, or mixed)."""

    key = "synergy_h1"
    category = "ingest"
    ConfigModel = SynergyH1UnifiedCfg

    @classmethod
    def input_contracts(cls):
        return {"raw?": "none"}  # file is optional; we can auto-discover

    @classmethod
    def output_contracts(cls):
        return {"df": "tidy.v1"}

    # ---------- helpers ----------

    def _auto_pick_one(self, files: list[object], mode: str) -> object:
        if mode == "single":
            if len(files) != 1:
                raise ParseError(
                    "Auto-discovery expected exactly one workbook, found "
                    f"{len(files)}:\n- "
                    + "\n- ".join(str(p) for p in files)
                    + "\nHint: set auto_pick: latest or auto_pick: merge, or pass reads.raw explicitly."
                )
            return files[0]
        if mode == "latest":
            return max(files, key=lambda p: p.stat().st_mtime)
        raise ParseError(f"_auto_pick_one called with mode={mode!r}")

    def _discover(self, ctx, cfg: SynergyH1UnifiedCfg) -> list:
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
                f"No raw .xlsx discovered under {roots} (include={cfg.auto_include}, exclude={cfg.auto_exclude}).\n"
                "Hint: put raw files under ./inputs (default), or set auto_roots / reads.raw explicitly."
            )
        if cfg.auto_pick in ("single", "latest"):
            return [self._auto_pick_one(files, cfg.auto_pick)]
        if cfg.auto_pick == "merge":
            return files
        raise ParseError(f"Unknown auto_pick mode {cfg.auto_pick!r} (expected: single|latest|merge)")

    def _log_file_overview(self, ctx, path, sheet_names: Sequence[str] | None):
        try:
            xl = pd.ExcelFile(path)
            total = len(xl.sheet_names)
            sel = len(sheet_names or xl.sheet_names)
            shown = (sheet_names or xl.sheet_names)[:5]
            extra = "" if sel <= 5 else " …"
            ctx.logger.debug(
                f"[accent]{path.name}[/accent] • sheets: {total} • selected: {sel} "
                f"({', '.join(map(str, shown))}{extra})"
            )
        except Exception:
            ctx.logger.debug(f"[accent]{path.name}[/accent] • sheets: ? (failed to introspect)")

    def _log_df_summary(self, ctx, df: pd.DataFrame, files_count: int, mode: str):
        try:
            n_rows = len(df)
            n_pos = df["position"].nunique()
            chans = sorted(df["channel"].astype(str).unique().tolist())
            tmin = float(pd.to_numeric(df["time"]).min()) if n_rows else 0.0
            tmax = float(pd.to_numeric(df["time"]).max()) if n_rows else 0.0
            src_counts = df["source"].value_counts().to_dict() if "source" in df.columns else {}
            sheets = int(df["sheet_name"].nunique()) if "sheet_name" in df.columns else 0
            ctx.logger.info(
                "Synergy H1 ingest • files=%d • mode=[bold]%s[/bold] • rows=%d • positions=%d • channels=%d "
                "• time=[%.2f, %.2f] h • sheets=%d • parts=%s",
                files_count,
                mode,
                n_rows,
                n_pos,
                len(chans),
                tmin,
                tmax,
                sheets,
                src_counts or "{}",
            )
            if chans:
                preview = ", ".join(chans[:8]) + (" …" if len(chans) > 8 else "")
                ctx.logger.debug("channels: %s", preview)

            # Block presence line (unambiguous snapshot/kinetic summary)
            snap_n = int(src_counts.get("snapshot", 0))
            kin_n = int(src_counts.get("kinetic", 0))
            snap_flag = "YES" if snap_n > 0 else "NO"
            kin_flag = "YES" if kin_n > 0 else "NO"
            ctx.logger.debug(
                "parsed blocks • snapshot=%s (%d rows) • kinetic=%s (%d rows)", snap_flag, snap_n, kin_flag, kin_n
            )

            # Duplicate key check (sanity): (position, channel, time) should be unique
            dups = int(df.duplicated(subset=["position", "channel", "time"], keep=False).sum()) if n_rows else 0
            if dups:
                ctx.logger.warning(
                    "[warn]duplicates detected[/warn] • (position,channel,time) duplicate rows: %d", dups
                )
        except Exception:
            # Logging should never break the pipeline
            pass

    # ---------- run ----------

    def run(self, ctx, inputs, cfg: SynergyH1UnifiedCfg):
        effective_mode = cfg.mode

        try:
            files = [inputs["raw"]] if "raw" in inputs else self._discover(ctx, cfg)

            if cfg.print_summary:
                file_names = ", ".join(getattr(f, "name", str(f)) for f in files)
                ctx.logger.info(
                    "[muted]Synergy H1 ingest • %d file(s) selected[/muted]%s",
                    len(files),
                    f" • {file_names}" if file_names else "",
                )
                for f in files:
                    self._log_file_overview(ctx, f, cfg.sheet_names)

            frames: list[pd.DataFrame] = []
            for f in files:
                if effective_mode == "kinetic_only":
                    df = parse_kinetic_only(
                        f,
                        channels=cfg.channels,
                        channel_map=cfg.channel_map,
                        sheet_names=cfg.sheet_names,
                        add_sheet=cfg.add_sheet,
                    )
                elif effective_mode == "snapshot_only":
                    df = parse_snapshot_and_timeseries(
                        f,
                        channels=cfg.channels,
                        channel_map=cfg.channel_map,
                        sheet_names=cfg.sheet_names,
                        add_sheet=cfg.add_sheet,
                        include_snapshot=True,
                        include_kinetic=False,
                    )
                elif effective_mode in ("mixed", "auto"):
                    # Mixed/auto: parse both sections if present; if parsing fails, surface the error (no fallback).
                    df = parse_snapshot_and_timeseries(
                        f,
                        channels=cfg.channels,
                        channel_map=cfg.channel_map,
                        sheet_names=cfg.sheet_names,
                        add_sheet=cfg.add_sheet,
                        include_snapshot=True,
                        include_kinetic=True,
                    )
                else:
                    raise ParseError(f"Unknown mode {effective_mode!r}")

                if cfg.add_source_column:
                    df = df.copy()
                    df[cfg.source_col] = getattr(f, "name", str(f))
                frames.append(df)

            if not frames:
                raise ParseError("No frames parsed from selected files")
            out = pd.concat(frames, ignore_index=True)

            if cfg.print_summary:
                self._log_df_summary(ctx, out, files_count=len(files), mode=effective_mode)

        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Synergy H1 ingest failed: {e}") from e

        return {"df": out}
