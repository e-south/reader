"""
SFXI triptych sequence figure assembly.

Reader owns the plate-reader figure composition. Sequence rendering and
artifact publication are delegated to narrow sibling adapters.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from reader.domains.logic.sfxi.triptych_sequence_outputs import (
    TRIPTYCH_BUNDLE_CONTRACT_VERSION,
)
from reader.domains.logic.sfxi.triptych_sequence_outputs import (
    bundle_paths as _bundle_paths,
)
from reader.domains.logic.sfxi.triptych_sequence_outputs import (
    cleanup_staging_root as _cleanup_staging_root,
)
from reader.domains.logic.sfxi.triptych_sequence_outputs import (
    manifest_payload as _manifest_payload,
)
from reader.domains.logic.sfxi.triptych_sequence_outputs import (
    publish_bundle as _publish_bundle,
)
from reader.domains.logic.sfxi.triptych_sequence_outputs import (
    relative_to_outputs as _relative_to_outputs,
)
from reader.domains.logic.sfxi.triptych_sequence_outputs import (
    staging_parent as _staging_parent,
)
from reader.domains.logic.sfxi.triptych_sequence_outputs import (
    staging_paths as _staging_paths,
)
from reader.domains.logic.sfxi.triptych_sequence_outputs import (
    write_movie as _write_movie,
)
from reader.domains.plate_reader.analysis.timepoints import infer_acquisition_transition_time_h
from reader.domains.plate_reader.plots.panels import (
    draw_time_series_panel,
    marker_map_for_levels,
    select_snapshot_rows,
    summarize_snapshot_values,
)
from reader.domains.promoter.candidate_bindings import (
    PromoterCandidateBinding,
    PromoterCandidateBindings,
    load_promoter_candidate_bindings,
)
from reader.domains.promoter.sequence_panel import (
    PromoterSequencePanelError,
    render_candidate_sequence_panel,
    require_baserender_api,
)
from reader.errors import SFXIError

STATE_ORDER = ("00", "10", "01", "11")
STATE_COLORS = {
    "00": "#8E8E8E",
    "10": "#4C78A8",
    "01": "#F58518",
    "11": "#54A24B",
}
_STATE_COLUMN = "__reader_sfxi_state"


def render_sfxi_triptych_sequence_bundle(
    *,
    ctx,
    vec8: pd.DataFrame,
    assay: pd.DataFrame,
    candidate_bindings_manifest: Path,
    config: Mapping[str, Any],
) -> list[Path]:
    try:
        require_baserender_api()
        bindings = load_promoter_candidate_bindings(candidate_bindings_manifest)
    except (FileNotFoundError, ValueError, PromoterSequencePanelError) as exc:
        raise SFXIError(f"SFXI triptych candidate-binding dependency is invalid: {exc}") from exc
    cfg = _normalize_config(config)
    _require_columns(vec8, ["design_id"], where="sfxi triptych vec8")
    _require_columns(
        assay,
        [cfg["design_col"], cfg["time_col"], "channel", "value", cfg["treatment_column"]],
        where="sfxi triptych assay",
    )
    assay = _bind_treatment_states(assay=assay, cfg=cfg)
    plan = _build_candidate_plan(vec8=vec8, bindings=bindings, cfg=cfg)
    if plan.empty:
        raise SFXIError("SFXI triptych sequence has no candidate rows to render.")
    if cfg["limit"] is not None:
        plan = plan.iloc[: int(cfg["limit"])].copy()

    scales = _compute_render_scales(assay=assay, render_plan=plan, cfg=cfg)
    bundle_id = _slug(cfg["bundle_id"])
    final = _bundle_paths(ctx=ctx, bundle_id=bundle_id)
    staging_root = Path(mkdtemp(prefix=f"{bundle_id}__", dir=str(_staging_parent(ctx.outputs_dir))))
    staging = _staging_paths(staging_root=staging_root, bundle_id=bundle_id, movie_enabled=cfg["movie_enabled"])
    try:
        records: list[dict[str, Any]] = []
        with PdfPages(staging["pdf"]) as pdf:
            for row_number, (_, row) in enumerate(plan.iterrows(), start=1):
                fig, record = _render_one(
                    assay=assay,
                    row=row,
                    cfg=cfg,
                    scales=scales,
                )
                png_path = staging["frames_dir"] / _frame_filename(
                    row_number=row_number, display_label=row["display_label"]
                )
                final_png_path = final["frames_dir"] / png_path.name
                fig.savefig(png_path, dpi=cfg["dpi"], facecolor="white")
                pdf.savefig(fig, facecolor="white")
                if len(records) == 0:
                    fig.savefig(staging["poster"], dpi=cfg["dpi"], facecolor="white")
                _close_figure(fig)
                records.append(
                    {**record, "png_path": _relative_to_outputs(final_png_path, outputs_dir=ctx.outputs_dir)}
                )

        movie_path = _write_movie(cfg=cfg, records=records, staging=staging, outputs_dir=ctx.outputs_dir)
        index = pd.DataFrame(records)
        index.to_csv(staging["index"], index=False)
        manifest = _manifest_payload(
            ctx=ctx,
            cfg=cfg,
            records=records,
            outputs=final,
            movie_path=final.get("movie") if movie_path is not None else None,
            scales=scales,
            bindings=bindings,
        )
        staging["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        _cleanup_staging_root(staging_root)
        raise
    _publish_bundle(staging=staging, final=final)

    paths = [final["poster"], final["pdf"], final["index"], final["manifest"]]
    if cfg["movie_enabled"]:
        paths.append(final["movie"])
    paths.extend(sorted(final["frames_dir"].glob("*.png")))
    return paths


def _normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config or {})
    removed_keys = sorted({"treatment_col", "treatments"} & set(cfg))
    if removed_keys:
        raise SFXIError(
            "SFXI triptych treatment identity comes from the resolved logic-map contract; "
            f"remove duplicated setting(s): {removed_keys}"
        )
    channels = dict(cfg.get("channels") or {})
    sequence_panel = dict(cfg.get("sequence_panel") or {})
    time_series = dict(cfg.get("time_series") or {})
    axis_limits = dict(cfg.get("axis_limits") or {})
    logic_map_ref = _required_string(cfg.get("logic_map_ref"), where="logic_map_ref")
    treatment_column = _required_string(cfg.get("treatment_column"), where="treatment_column")
    treatment_map, treatment_case_sensitive = _normalize_treatment_contract(
        cfg.get("treatment_map"),
        case_sensitive=cfg.get("treatment_case_sensitive"),
    )
    return {
        "logic_map_ref": logic_map_ref,
        "bundle_id": str(cfg.get("bundle_id") or "sfxi_triptych_sequence"),
        "design_col": str(cfg.get("design_col") or "design_id"),
        "sequence_col": str(cfg.get("sequence_col") or "sequence"),
        "time_col": str(cfg.get("time_col") or "time"),
        "treatment_column": treatment_column,
        "treatment_map": treatment_map,
        "treatment_case_sensitive": treatment_case_sensitive,
        "state_column": _STATE_COLUMN,
        "snapshot_target_time_h": _finite_float(cfg.get("snapshot_target_time_h"), default=12.0),
        "acquisition_transition_time_h": _finite_float(cfg.get("acquisition_transition_time_h"), default=None),
        "time_tolerance_h": _finite_float(cfg.get("time_tolerance_h"), default=0.51),
        "limit": cfg.get("limit"),
        "dpi": int(cfg.get("dpi") or 220),
        "movie_enabled": bool(cfg.get("movie_enabled", False)),
        "movie_fps": float(cfg.get("movie_fps", 0.85)),
        "channels": {
            "growth": str(channels.get("growth") or "OD600"),
            "ratio": str(channels.get("ratio") or "YFP/CFP"),
            "snapshot": str(channels.get("snapshot") or channels.get("ratio") or "YFP/CFP"),
        },
        "sequence_panel": {
            "profile": str(sequence_panel.get("profile") or "promoter_compact_slide.v1"),
            "target_width_px": int(sequence_panel.get("target_width_px") or 2200),
            "target_height_px": int(sequence_panel.get("target_height_px") or 310),
            "vertical_anchor": str(sequence_panel.get("vertical_anchor") or "center"),
            "canvas_top_pad_px": int(sequence_panel.get("canvas_top_pad_px") or 0),
            "style_overrides": dict(sequence_panel.get("style_overrides") or {}),
        },
        "time_series": time_series,
        "axis_limits": axis_limits,
        "states": [
            {
                "state": state,
                "label": treatment_map[state],
                "short_label": state,
                "color": STATE_COLORS[state],
            }
            for state in STATE_ORDER
        ],
    }


def _bind_treatment_states(*, assay: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    state_column = str(cfg["state_column"])
    if state_column in assay.columns:
        raise SFXIError(f"SFXI triptych assay uses reserved internal column {state_column!r}.")
    treatment_column = str(cfg["treatment_column"])
    treatment_map = dict(cfg["treatment_map"])
    case_sensitive = bool(cfg["treatment_case_sensitive"])
    reverse = {(label if case_sensitive else label.strip().casefold()): state for state, label in treatment_map.items()}
    out = assay.copy()
    treatment_values = out[treatment_column].astype(str)
    if not case_sensitive:
        treatment_values = treatment_values.str.strip().str.casefold()
    out[state_column] = treatment_values.map(reverse)
    return out


def _frame_filename(*, row_number: int, display_label: object) -> str:
    return f"{row_number:03d}_{_slug(display_label)}.png"


def _build_candidate_plan(
    *,
    vec8: pd.DataFrame,
    bindings: PromoterCandidateBindings,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    design_col = str(cfg["design_col"])
    vec = vec8.copy()
    vec[design_col] = vec[design_col].astype(str)
    if vec[design_col].duplicated().any():
        dupes = sorted(vec.loc[vec[design_col].duplicated(), design_col].astype(str).unique())
        raise SFXIError(f"SFXI triptych requires one vec8 row per design; duplicates: {dupes}")

    binding_by_design = {binding.reader_design_id: binding for binding in bindings.rows}
    missing = sorted(set(vec[design_col]) - set(binding_by_design))
    if missing:
        raise SFXIError(f"SFXI triptych designs are absent from the exact candidate-binding artifact: {missing}")
    plan = vec.copy()
    plan["candidate_binding"] = plan[design_col].map(binding_by_design)
    sequence_col = str(cfg["sequence_col"])
    if sequence_col in plan.columns:
        mismatches = [
            str(row[design_col])
            for _, row in plan.iterrows()
            if str(row[sequence_col]).upper() != row["candidate_binding"].canonical_sequence.upper()
        ]
        if mismatches:
            raise SFXIError(
                "SFXI vec8 sequence disagrees with the study-issued candidate binding for designs: "
                f"{sorted(mismatches)}"
            )
    plan["display_label"] = plan["candidate_binding"].map(lambda binding: binding.display_label)
    plan["row_kind"] = "candidate"
    plan["score_status"] = "canonical_sfxi_vec8"
    plan["snapshot_time_h"] = float(cfg["snapshot_target_time_h"])
    plan["snapshot_time_source"] = "sfxi_triptych_sequence.snapshot_target_time_h"
    if "time_selected_h" in plan.columns:
        plan["vec8_time_selected_h"] = pd.to_numeric(plan["time_selected_h"], errors="coerce")
    else:
        plan["vec8_time_selected_h"] = math.nan
    return _sort_designs(plan, design_col=design_col)


def _render_one(*, assay: pd.DataFrame, row: pd.Series, cfg: Mapping[str, Any], scales: Mapping[str, Any]):
    design_col = str(cfg["design_col"])
    design_id = str(row[design_col])
    assay_design = assay[assay[design_col].astype(str) == design_id].copy()
    if assay_design.empty:
        raise SFXIError(f"No assay rows available for design_id={design_id!r}.")
    assay_design["plot_time_h"] = pd.to_numeric(assay_design[cfg["time_col"]], errors="coerce")
    assay_design["value"] = pd.to_numeric(assay_design["value"], errors="coerce")
    assay_design = assay_design.dropna(subset=["plot_time_h", "value", "channel", cfg["state_column"]])

    states = list(cfg["states"])
    treatment_order = [item["state"] for item in states]
    color_map = {item["state"]: item["color"] for item in states}
    label_map = {item["state"]: item["label"] for item in states}
    marker_map = marker_map_for_levels(treatment_order)
    time_meta = _time_metadata(assay_design, cfg=cfg)

    fig = plt.figure(figsize=(10.7, 5.82), dpi=180)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.43], hspace=0.20, wspace=0.25)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    seq_ax = fig.add_subplot(gs[1, :])
    for ax in axes:
        ax.set_box_aspect(1)

    snapshot_meta = _draw_snapshot_panel(
        axes[2],
        assay=assay_design,
        channel=str(cfg["channels"]["snapshot"]),
        snapshot_time_h=float(row["snapshot_time_h"]),
        tolerance_h=float(cfg["time_tolerance_h"]),
        states=states,
        cfg=cfg,
        y_limits=scales["y_limits"],
    )
    time_meta["snapshot_display_time_h"] = float(snapshot_meta["time_used_h"])
    x_limits = _compute_row_x_limits(assay=assay_design, time_meta=time_meta, cfg=cfg)
    for ax, channel, ylabel, title, event_labels in (
        (axes[0], str(cfg["channels"]["growth"]), "OD$_{600}$", "Growth", False),
        (axes[1], str(cfg["channels"]["ratio"]), "YFP/CFP", "Reporter ratio", True),
    ):
        _draw_time_panel(
            ax,
            assay=assay_design,
            channel=channel,
            ylabel=ylabel,
            title=title,
            treatment_order=treatment_order,
            color_map=color_map,
            marker_map=marker_map,
            label_map=label_map,
            time_meta=time_meta,
            y_limits=scales["y_limits"],
            x_limits=x_limits,
            cfg=cfg,
            show_event_labels=event_labels,
        )

    binding = row["candidate_binding"]
    if not isinstance(binding, PromoterCandidateBinding):
        raise SFXIError("SFXI triptych render plan is missing its typed candidate binding.")
    panel = cfg["sequence_panel"]
    try:
        rendered = render_candidate_sequence_panel(
            binding,
            style_profile=str(panel["profile"]),
            style_overrides=dict(panel["style_overrides"]),
            target_width_px=int(panel["target_width_px"]),
            target_height_px=int(panel["target_height_px"]),
            vertical_anchor=str(panel["vertical_anchor"]),
            canvas_top_pad_px=int(panel["canvas_top_pad_px"]),
        )
    except PromoterSequencePanelError as exc:
        raise SFXIError(f"Could not render bound promoter sequence: {exc}") from exc
    seq_ax.imshow(rendered.image)
    seq_ax.set_axis_off()
    diagnostics = rendered.diagnostics
    fig.suptitle(str(row["display_label"]), x=0.5, y=0.984, ha="center", fontsize=19.0, fontweight="semibold")
    trajectory_ci = float((cfg.get("time_series") or {}).get("ci", 95.0))
    fig.text(
        0.5,
        0.934,
        f"Time series: mean with {trajectory_ci:g}% bootstrap CI  ·  Snapshot: observed wells, mean, and sample SD",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#475569",
    )
    fig.subplots_adjust(left=0.062, right=0.988, bottom=0.060, top=0.902)

    record = {
        "design_id": design_id,
        "display_label": str(row["display_label"]),
        "row_kind": str(row["row_kind"]),
        "score_status": str(row["score_status"]),
        "candidate_id": binding.candidate_id,
        "sequence_sha256": binding.sequence_sha256,
        "sequence_authority_dataset_id": binding.sequence_authority_dataset_id,
        "sequence_authority_id": binding.sequence_authority_id,
        "sequence_authority_sha256": binding.sequence_authority_sha256,
        "sequence_adapter_kind": binding.baserender_adapter_kind,
        "snapshot_target_time_h": float(row["snapshot_time_h"]),
        "snapshot_time_source": str(row.get("snapshot_time_source", "unknown")),
        "snapshot_observed_time_h": float(snapshot_meta["time_used_h"]),
        "snapshot_fell_back": bool(snapshot_meta["fell_back"]),
        "snapshot_fallback_delta_h": snapshot_meta.get("fallback_delta_h"),
        "vec8_selected_time_h": _json_float_or_none(row.get("vec8_time_selected_h")),
        "acquisition_transition_time_h": _json_float_or_none(time_meta.get("acquisition_transition_display_time_h")),
        "x_min_h": x_limits[0],
        "x_max_h": x_limits[1],
        "sequence_panel": asdict(diagnostics),
    }
    return fig, record


def _time_metadata(assay: pd.DataFrame, *, cfg: Mapping[str, Any]) -> dict[str, float]:
    configured = cfg.get("acquisition_transition_time_h")
    transition = (
        float(configured)
        if configured is not None
        else infer_acquisition_transition_time_h(assay, time_col="plot_time_h")
    )
    return {
        "acquisition_transition_display_time_h": float(transition) if transition is not None else math.nan,
        "snapshot_display_time_h": float(cfg["snapshot_target_time_h"]),
    }


def _draw_time_panel(
    ax,
    *,
    assay: pd.DataFrame,
    channel: str,
    ylabel: str,
    title: str,
    treatment_order: list[str],
    color_map: dict[str, str],
    marker_map: dict[str, str],
    label_map: dict[str, str],
    time_meta: dict[str, float],
    y_limits: dict[str, list[float]],
    x_limits: list[float],
    cfg: Mapping[str, Any],
    show_event_labels: bool = False,
) -> None:
    treatment_col = str(cfg["state_column"])
    data = assay[assay["channel"].astype(str) == channel].copy()
    observed = set(data[treatment_col].astype(str))
    missing = [label for label in treatment_order if label not in observed]
    if missing:
        raise SFXIError(f"{channel} panel missing SFXI states: {_describe_states(missing, cfg=cfg)}")
    data = data[data[treatment_col].astype(str).isin(treatment_order)].copy()
    segment_col = _add_segment_column(data)
    ts_cfg = dict(cfg.get("time_series") or {})
    draw_time_series_panel(
        ax,
        data=data,
        x_col="plot_time_h",
        hue_col=treatment_col,
        hue_levels=treatment_order,
        color_map=color_map,
        marker_map=marker_map,
        segment_col=segment_col,
        show_replicates=bool(ts_cfg.get("show_replicates", False)),
        ci=float(ts_cfg.get("ci", 95.0)),
        ci_alpha=float(ts_cfg.get("ci_alpha", 0.16)),
        ci_boot=int(ts_cfg.get("ci_boot", 300)),
        ci_seed=0,
        line_alpha=0.92,
        mean_marker_alpha=0.86,
        replicate_alpha=0.18,
        add_sheet_lines=math.isfinite(float(time_meta["acquisition_transition_display_time_h"])),
        sheet_lines=[float(time_meta["acquisition_transition_display_time_h"])],
        sheet_line_kwargs={"color": "#5F5F5F", "linestyle": "--", "linewidth": 2.15, "alpha": 0.98, "zorder": 0.4},
        log_y=False,
        xlabel="Time (h)",
        ylabel=ylabel,
        legend_loc="upper left",
        show_legend=False,
        legend_label_map=label_map,
        marked_time=float(time_meta["snapshot_display_time_h"]),
        marked_time_kwargs={"color": "#5F5F5F", "linestyle": "--", "linewidth": 2.15, "alpha": 0.98, "zorder": 0.6},
        line_width=2.2,
        mean_marker_size=30.0,
        mean_marker_every=int(ts_cfg.get("mean_marker_every", 12)),
        axis_label_size=13.4,
        tick_label_size=12.3,
        legend_fontsize=8.5,
        legend_marker_size=5.8,
    )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits[channel])
    if show_event_labels:
        _draw_time_event_labels(ax, time_meta=time_meta, font_size=float(ts_cfg.get("event_label_font_size", 10.0)))
    ax.set_title(title, loc="center", fontsize=14.3, pad=7)
    _style_axis(ax)


def _draw_snapshot_panel(
    ax,
    *,
    assay: pd.DataFrame,
    channel: str,
    snapshot_time_h: float,
    tolerance_h: float,
    states: list[dict[str, str]],
    cfg: Mapping[str, Any],
    y_limits: dict[str, list[float]],
) -> dict[str, Any]:
    treatment_col = str(cfg["state_column"])
    key_cols = [str(cfg["design_col"]), treatment_col, "channel", "position"]
    key_cols = [column for column in key_cols if column in assay.columns]
    snapshot_df = assay.copy()
    snapshot_df["time"] = snapshot_df["plot_time_h"]
    selection = select_snapshot_rows(
        df=snapshot_df,
        target_time=float(snapshot_time_h),
        keys=key_cols,
        channel=channel,
        tolerance=float(tolerance_h),
    )
    if selection.rows.empty:
        raise SFXIError(f"No snapshot rows for channel={channel!r} near t={snapshot_time_h:.2f} h")
    stats = summarize_snapshot_values(df=selection.rows, group_cols=[treatment_col], err="sd")
    order = [item["state"] for item in states]
    stats = stats.set_index(treatment_col).reindex(order)
    missing = stats[stats["mean"].isna()].index.astype(str).tolist()
    if missing:
        raise SFXIError(f"Snapshot panel missing SFXI states: {_describe_states(missing, cfg=cfg)}")
    positions = np.arange(len(order), dtype=float)
    means = stats["mean"].astype(float).to_numpy()
    sd = stats["std"].astype(float).fillna(0.0).to_numpy()
    colors = [item["color"] for item in states]
    ax.vlines(positions, means - sd, means + sd, color="#475569", linewidth=1.2, zorder=2)
    ax.scatter(positions, means - sd, marker="_", s=52, color="#475569", linewidths=1.2, zorder=2)
    ax.scatter(positions, means + sd, marker="_", s=52, color="#475569", linewidths=1.2, zorder=2)
    for idx, treatment in enumerate(order):
        points = selection.rows[selection.rows[treatment_col].astype(str) == treatment]
        jitter = np.linspace(-0.09, 0.09, len(points)) if len(points) > 1 else np.asarray([0.0])
        ax.scatter(
            positions[idx] + jitter,
            points["value"].astype(float),
            s=18,
            facecolors="white",
            edgecolors="#94A3B8",
            linewidths=0.8,
            alpha=0.92,
            zorder=3,
        )
        ax.hlines(means[idx], positions[idx] - 0.18, positions[idx] + 0.18, color=colors[idx], linewidth=2.4, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([_snapshot_tick_label(item["short_label"]) for item in states], ha="center", fontsize=11.5)
    ax.set_xlabel("")
    ax.set_ylabel(channel, fontsize=13.4)
    ax.set_ylim(y_limits[channel])
    ax.set_title(f"Snapshot ({float(selection.time_used):.1f} h)", loc="center", fontsize=14.3, pad=7)
    ax.tick_params(axis="y", labelsize=12.3)
    ax.yaxis.grid(True, which="major", color="#E1E1E1", linewidth=0.75)
    ax.xaxis.grid(False)
    _style_axis(ax)
    return {
        "time_used_h": float(selection.time_used),
        "fell_back": bool(selection.fell_back),
        "fallback_delta_h": selection.fallback_delta,
    }


def _compute_render_scales(*, assay: pd.DataFrame, render_plan: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    values_by_channel: dict[str, list[float]] = {
        str(cfg["channels"]["growth"]): [],
        str(cfg["channels"]["ratio"]): [],
        str(cfg["channels"]["snapshot"]): [],
    }
    treatment_states = {item["state"] for item in cfg["states"]}
    treatment_col = str(cfg["state_column"])
    design_col = str(cfg["design_col"])
    design_ids = set(render_plan[design_col].astype(str))
    sub = assay[assay[design_col].astype(str).isin(design_ids)].copy()
    for channel, values in values_by_channel.items():
        channel_rows = sub[
            (sub["channel"].astype(str) == channel) & sub[treatment_col].astype(str).isin(treatment_states)
        ]
        values.extend(pd.to_numeric(channel_rows["value"], errors="coerce").dropna().astype(float).tolist())
        if channel == str(cfg["channels"]["snapshot"]):
            snapshot_stats = (
                channel_rows.assign(value=pd.to_numeric(channel_rows["value"], errors="coerce"))
                .dropna(subset=["value"])
                .groupby([design_col, treatment_col, str(cfg["time_col"])], dropna=False)["value"]
                .agg(["mean", "std"])
            )
            whisker_upper = snapshot_stats["mean"] + snapshot_stats["std"].fillna(0.0)
            values.extend(whisker_upper.astype(float).tolist())
    axis_cfg = dict(cfg.get("axis_limits") or {})
    pad_fraction = float(axis_cfg.get("y_padding_fraction", 0.08))
    upper_quantile = float(axis_cfg.get("upper_quantile", 1.0))
    if not 0 < upper_quantile <= 1:
        raise SFXIError("axis_limits.upper_quantile must be > 0 and <= 1.")
    y_limits: dict[str, list[float]] = {}
    for channel, values in values_by_channel.items():
        if not values:
            raise SFXIError(f"No values available for global y-axis scaling: {channel}")
        upper_source = float(np.quantile(np.asarray(values, dtype=float), upper_quantile))
        upper = max(upper_source * (1.0 + pad_fraction), 1e-9)
        y_limits[channel] = [0.0, _nice_upper(upper)]
    return {"x_policy": "per_row_raw_assay_time", "y_limits": y_limits}


def _compute_row_x_limits(
    *, assay: pd.DataFrame, time_meta: Mapping[str, float], cfg: Mapping[str, Any]
) -> list[float]:
    values = pd.to_numeric(assay["plot_time_h"], errors="coerce").dropna().astype(float).tolist()
    for key in ("acquisition_transition_display_time_h", "snapshot_display_time_h"):
        value = float(time_meta[key])
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise SFXIError("Cannot compute x-axis limits: no finite time values.")
    x_pad = float((cfg.get("time_axis") or {}).get("x_padding_h", 0.5))
    return [float(min(0.0, min(values))), _nice_upper(max(values) + max(0.0, x_pad))]


def _normalize_treatment_contract(
    raw_map: Any,
    *,
    case_sensitive: Any,
) -> tuple[dict[str, str], bool]:
    if not isinstance(raw_map, Mapping) or set(raw_map) != set(STATE_ORDER):
        raise SFXIError("SFXI triptych treatment_map must have exactly the states 00, 10, 01, and 11.")
    treatment_map = {state: _required_string(raw_map[state], where=f"treatment_map.{state}") for state in STATE_ORDER}
    if not isinstance(case_sensitive, bool):
        raise SFXIError("SFXI triptych treatment_case_sensitive must be true or false.")
    labels = list(treatment_map.values())
    normalized = labels if case_sensitive else [label.strip().casefold() for label in labels]
    if len(set(normalized)) != len(STATE_ORDER):
        raise SFXIError("SFXI triptych treatment_map labels must be unique under its case-sensitivity policy.")
    return treatment_map, case_sensitive


def _required_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SFXIError(f"SFXI triptych {where} must be a non-empty string.")
    return value.strip()


def _describe_states(states: Sequence[str], *, cfg: Mapping[str, Any]) -> list[str]:
    treatment_map = dict(cfg["treatment_map"])
    return [f"{state}={treatment_map[state]!r}" for state in states]


def _draw_time_event_labels(ax, *, time_meta: Mapping[str, float], font_size: float) -> None:
    transform = ax.get_xaxis_transform()
    for label, value, x_offset in (
        ("acquisition transition", time_meta.get("acquisition_transition_display_time_h"), -12),
        ("snapshot", time_meta.get("snapshot_display_time_h"), 12),
    ):
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x):
            continue
        ax.annotate(
            label,
            xy=(x, 0.5),
            xycoords=transform,
            xytext=(x_offset, 0),
            textcoords="offset points",
            clip_on=True,
            va="center",
            ha="center",
            fontsize=font_size,
            color="#555555",
            rotation=90,
            rotation_mode="anchor",
            zorder=4.5,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.74},
        )


def _add_segment_column(data: pd.DataFrame) -> str | None:
    segment_parts = [column for column in ("source", "sheet_name", "sheet_index") if column in data.columns]
    if not segment_parts:
        return None
    segment_col = "__plot_segment"
    segments = data[segment_parts].copy()
    for column in segment_parts:
        segments[column] = segments[column].astype(str)
    data[segment_col] = segments.agg("::".join, axis=1)
    return segment_col


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#868686")
    ax.spines["bottom"].set_color("#868686")
    ax.spines["left"].set_linewidth(1.45)
    ax.spines["bottom"].set_linewidth(1.45)
    ax.tick_params(colors="#2F2F2F", width=1.15)
    ax.yaxis.grid(True, which="major", color="#E1E1E1", linewidth=0.75)
    ax.xaxis.grid(True, which="major", color="#ECECEC", linewidth=0.65)


def _snapshot_tick_label(label: str) -> str:
    return str(label).replace(" ", "\n", 1)


def _display_label(design_id: object) -> str:
    text = str(design_id)
    prefix = "pDual-10-"
    return text[len(prefix) :] if text.startswith(prefix) else text


def _sort_designs(df: pd.DataFrame, *, design_col: str) -> pd.DataFrame:
    out = df.copy()
    out["__design_sort"] = out[design_col].map(_design_sort_key)
    out = out.sort_values("__design_sort", kind="stable").drop(columns=["__design_sort"])
    return out.reset_index(drop=True)


def _design_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 10**9, text)


def _slug(value: object) -> str:
    text = str(value)
    keep = [ch.lower() if ch.isalnum() else "_" for ch in text]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "item"


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _finite_float(value: Any, *, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SFXIError(f"Expected a finite numeric value, got {value!r}") from exc
    if not math.isfinite(out):
        raise SFXIError(f"Expected a finite numeric value, got {value!r}")
    return out


def _json_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _nice_upper(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        return 1.0
    if value <= 1.5:
        step = 0.1
    elif value <= 5:
        step = 0.25
    else:
        step = 1.0
    return float(math.ceil(value / step) * step)


def _close_figure(fig) -> None:
    plt.close(fig)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, where: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SFXIError(f"{where}: missing required columns {missing}")


__all__ = [
    "TRIPTYCH_BUNDLE_CONTRACT_VERSION",
    "render_sfxi_triptych_sequence_bundle",
]
