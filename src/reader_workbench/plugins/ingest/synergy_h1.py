from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

import pandas as pd
from pydantic import Field, model_validator

from reader_workbench.domains.plate_reader.io.synergy_h1 import (
    parse_kinetic_only,
    parse_snapshot_and_timeseries,
    probe_synergy_workbook,
)
from reader_workbench.errors import ParseError
from reader_workbench.plugins.ingest._discovery import discover_auto_input_files
from reader_workbench.plugins.ingest.discovery_policy import DEFAULT_EXCLUDE, DEFAULT_INCLUDE
from reader_workbench.workbench.ports import dataframe_output, file_path_input
from reader_workbench.workbench.registry import Plugin, PluginConfig, PreflightIssue


class SynergyH1UnifiedCfg(PluginConfig):
    # What kind of ingest are we doing?
    mode: Literal["snapshot_only", "kinetic_only", "mixed"] = "mixed"

    # parsing knobs
    channels: list[str] | None = None
    channel_map: Mapping[str, str] | None = None
    sheet_names: Sequence[str] | None = None

    # time normalization
    time_round_decimals: int | None = 12
    time_step_h: float | None = None

    # auto-discovery knobs
    auto_roots: list[str] | None = None
    auto_include: list[str] = Field(default_factory=lambda: list(DEFAULT_INCLUDE))
    auto_exclude: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    auto_pick: Literal["single", "latest"] = "single"
    auto_recursive: bool = False

    # logging
    print_summary: bool = True

    @model_validator(mode="after")
    def require_channel_contract(self) -> Self:
        if self.mode in {"snapshot_only", "mixed"} and not self.channel_map:
            raise ValueError(f"Synergy H1 mode {self.mode!r} requires an explicit channel_map")
        if not self.channels and not self.channel_map:
            raise ValueError("Synergy H1 ingest requires channels or channel_map")
        return self


class SynergyH1(Plugin):
    """Unified Synergy H1 ingest (snapshot-only, kinetic-only, or mixed)."""

    ConfigModel = SynergyH1UnifiedCfg

    @classmethod
    def input_ports(cls):
        return {"raw": file_path_input("raw", optional=True)}

    @classmethod
    def output_ports(cls):
        return {"df": dataframe_output("df", "tidy.v1")}

    @classmethod
    def _selected_workbooks(
        cls,
        *,
        exp_dir: Path,
        cfg: SynergyH1UnifiedCfg,
        inputs: Mapping[str, object],
    ) -> list[Path]:
        if "raw" in inputs:
            raw_value = inputs["raw"]
            raw_path = raw_value if isinstance(raw_value, Path) else getattr(raw_value, "path", None)
            if raw_path is None:
                raise ParseError("Synergy H1 raw input must select a workbook file")
            path = Path(raw_path)
            return [path if path.is_absolute() else exp_dir / path]
        return discover_auto_input_files(
            exp_dir=exp_dir,
            auto_roots=cfg.auto_roots,
            auto_include=cfg.auto_include,
            auto_exclude=cfg.auto_exclude,
            auto_recursive=cfg.auto_recursive,
            auto_pick=cfg.auto_pick,
            discovery_label="raw .xlsx files",
            singular_label="workbook",
        )

    @classmethod
    def resolve_missing_file_inputs(cls, *, exp_dir, cfg: SynergyH1UnifiedCfg, inputs):
        if "raw" in inputs:
            return {}
        selected = cls._selected_workbooks(exp_dir=exp_dir, cfg=cfg, inputs=inputs)
        return {"raw": selected[0]}

    @classmethod
    def preflight_readiness(cls, *, exp_dir, cfg: SynergyH1UnifiedCfg, reads):
        try:
            selected = cls._selected_workbooks(exp_dir=exp_dir, cfg=cfg, inputs=reads)
        except ParseError as err:
            return (PreflightIssue(kind="file", message=str(err)),)

        issues: list[PreflightIssue] = []
        for path in selected:
            try:
                probe_synergy_workbook(path)
            except Exception as err:
                issues.append(
                    PreflightIssue(
                        kind="file",
                        message=f"Synergy H1 workbook {path} could not be opened: {err}",
                    )
                )
        return tuple(issues)

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

        except Exception:
            # Logging should never break the pipeline
            pass

    # ---------- run ----------

    def run(self, ctx, inputs, cfg: SynergyH1UnifiedCfg):
        effective_mode = cfg.mode

        try:
            raw_path = inputs.get("raw")
            if not isinstance(raw_path, Path):
                raise ParseError("Synergy H1 runtime requires one resolved raw workbook path")

            if cfg.print_summary:
                ctx.logger.info(
                    "[muted]Synergy H1 ingest • workbook selected[/muted] • %s",
                    raw_path.name,
                )
            if effective_mode == "kinetic_only":
                out = parse_kinetic_only(
                    raw_path,
                    channels=cfg.channels,
                    channel_map=cfg.channel_map,
                    sheet_names=cfg.sheet_names,
                    time_round_decimals=cfg.time_round_decimals,
                    time_step_h=cfg.time_step_h,
                )
            elif effective_mode == "snapshot_only":
                out = parse_snapshot_and_timeseries(
                    raw_path,
                    channels=cfg.channels,
                    channel_map=cfg.channel_map,
                    sheet_names=cfg.sheet_names,
                    time_round_decimals=cfg.time_round_decimals,
                    time_step_h=cfg.time_step_h,
                    include_snapshot=True,
                    include_kinetic=False,
                )
            elif effective_mode == "mixed":
                out = parse_snapshot_and_timeseries(
                    raw_path,
                    channels=cfg.channels,
                    channel_map=cfg.channel_map,
                    sheet_names=cfg.sheet_names,
                    time_round_decimals=cfg.time_round_decimals,
                    time_step_h=cfg.time_step_h,
                    include_snapshot=True,
                    include_kinetic=True,
                )
            else:
                raise ParseError(
                    f"Unsupported Synergy H1 mode {effective_mode!r}; expected snapshot_only, kinetic_only, or mixed"
                )

            if cfg.print_summary:
                self._log_df_summary(ctx, out, files_count=1, mode=effective_mode)

        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Synergy H1 ingest failed: {e}") from e

        return {"df": out}
