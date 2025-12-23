"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/logic_symmetry/main.py

Logic-symmetry plotter entrypoint.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .encodings import EncodingConfig, apply_encodings
from .extract_corners import MappingConfig, resolve_and_aggregate
from .io import save_plot, write_csv
from .metrics import CornerStats, compute_metrics
from .overlay import OverlayStyle
from .prep import prepare_for_logic_symmetry
from .render import VisualConfig, draw_scatter

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogicSymmetryResult:
    """Pure result of plot_logic_symmetry (no CSV read-back)."""

    table: pd.DataFrame  # typed metrics/encodings table (logic_symmetry.v1)
    fig: object  # matplotlib Figure
    plot_paths: list[Path]  # written plot files
    csv_path: Path | None = None  # optional companion CSV when enabled


def _dget(d: dict | None, key: str, default):
    if d is None:
        return default
    if key in d and d[key] is not None:
        return d[key]
    return default


def _pick_baseline_corner(row: pd.Series) -> str:
    order = ["b00", "b10", "b01", "b11"]
    vals = {k: float(row[k]) for k in order}
    m = min(vals.values())
    for k in order:
        if vals[k] == m:
            return {"b00": "00", "b10": "10", "b01": "01", "b11": "11"}[k]
    return "00"


def plot_logic_symmetry(
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir: str | Path,
    *,
    response_channel: str,
    design_by: list[str] = None,
    batch_col: str = "batch",
    treatment_map: dict[str, str] = None,
    treatment_case_sensitive: bool = True,
    aggregation: dict | None = None,
    encodings: dict | None = None,
    ideals_overlay: dict | None = None,
    visuals: dict | None = None,
    output: dict | None = None,
    prep: dict | None = None,
    fig_kwargs: dict | None = None,
    filename: str | None = None,
    subplots: str | None = None,
    iterate_genotypes: bool = False,
    palette_book=None,
    **_ignored,
) -> LogicSymmetryResult:
    if design_by is None:
        design_by = ["genotype"]
    if treatment_map is None or set(treatment_map.keys()) != {"00", "10", "01", "11"}:
        raise ValueError(
            "treatment_map must be provided with keys {'00','10','01','11'} and single exact labels as values"
        )

    replicate_stat = _dget(aggregation, "replicate_stat", "mean")
    if replicate_stat not in ("mean", "median"):
        raise ValueError(f"replicate_stat must be 'mean' or 'median', got {replicate_stat!r}")
    uncertainty_mode = _dget(aggregation, "uncertainty", "halo")
    if uncertainty_mode not in ("none", "errorbars", "halo"):
        raise ValueError(f"uncertainty must be one of 'none'|'errorbars'|'halo', got {uncertainty_mode!r}")

    enc_cfg = EncodingConfig(
        size_by=_dget(encodings, "size_by", "log_r"),
        size_fixed=float(_dget(encodings, "size_fixed", 80.0)),
        hue=_dget(encodings, "hue", None),
        alpha_by=_dget(encodings, "alpha_by", "batch"),
        alpha_min=float(_dget(encodings, "alpha_min", 0.35)),
        alpha_max=float(_dget(encodings, "alpha_max", 1.0)),
        shape_by=_dget(encodings, "shape_by", None),
        shape_cycle=list(_dget(encodings, "shape_cycle", ["o", "s", "^", "D", "P", "X", "v", "*"])),
        shape_max_categories=_dget(encodings, "shape_max_categories", None),
    )

    overlay_enable = bool(_dget(ideals_overlay, "enable", False))
    overlay_gate_set = _dget(ideals_overlay, "gate_set", "logic_family")
    overlay_style_cfg = _dget(ideals_overlay, "style", {})

    # Choose mode: explicit style.mode OR auto "tiles" for gate sets beginning with "tiles"
    mode_val = _dget(overlay_style_cfg, "mode", None)
    if mode_val is None:
        mode_val = "tiles" if str(overlay_gate_set).startswith("tiles") else "dot"

    overlay_style = OverlayStyle(
        mode=str(mode_val),
        alpha=float(_dget(overlay_style_cfg, "alpha", 0.25)),
        size=float(_dget(overlay_style_cfg, "size", 40.0)),
        face_color=str(_dget(overlay_style_cfg, "face_color", "#FFFFFF")),
        edge_color=str(_dget(overlay_style_cfg, "edge_color", _dget(overlay_style_cfg, "color", "#888888"))),
        show_labels=bool(_dget(overlay_style_cfg, "show_labels", True)),
        label_offset=float(_dget(overlay_style_cfg, "label_offset", 0.02)),
        label_line_height=float(_dget(overlay_style_cfg, "label_line_height", 0.018)),
        label_fontsize=int(_dget(overlay_style_cfg, "label_fontsize", 12)),  # NEW
        tile_cell_w=float(_dget(overlay_style_cfg, "tile_cell_w", 0.035)),
        tile_cell_h=float(_dget(overlay_style_cfg, "tile_cell_h", 0.035)),
        tile_gap=float(_dget(overlay_style_cfg, "tile_gap", 0.0)),
        tile_edge_width=float(_dget(overlay_style_cfg, "tile_edge_width", 0.6)),
        tiles_stack_multiple=bool(_dget(overlay_style_cfg, "tiles_stack_multiple", True)),
    )

    vis_cfg = VisualConfig(
        xlim=tuple(_dget(visuals, "xlim", (-1.02, 1.02))),
        ylim=tuple(_dget(visuals, "ylim", (-1.02, 1.02))),
        grid=bool(_dget(visuals, "grid", True)),
        color=str(_dget(visuals, "color", "#6e6e6e")),
        annotate_designs=bool(_dget(visuals, "annotate_designs", False)),
        design_label_col=_dget(visuals, "design_label_col", None) or (design_by[0] if design_by else None),
        label_fontsize=int(_dget(visuals, "label_fontsize", 12)),
        label_offset=float(_dget(visuals, "label_offset", 0.02)),
        axis_label_fontsize=int(_dget(visuals, "axis_label_fontsize", 16)),
        tick_label_fontsize=int(_dget(visuals, "tick_label_fontsize", 14)),
        title_fontsize=int(_dget(visuals, "title_fontsize", 18)),
        legend_fontsize=int(_dget(visuals, "legend_fontsize", 12)),
    )

    out_formats = [ext.lower() for ext in _dget(output, "format", ["pdf"])]
    dpi = int(_dget(output, "dpi", 300))
    figsize = tuple(_dget(output, "figsize", (7, 6)))

    base_name = filename or "logic_symmetry"

    LOG.info("logic_symmetry: starting")
    LOG.info("• response_channel=%s | design_by=%s | batch_col=%s", response_channel, design_by, batch_col)
    LOG.info("• replicate_stat=%s | uncertainty=%s", replicate_stat, uncertainty_mode)
    LOG.info(
        "• treatment_map: 00=%r | 10=%r | 01=%r | 11=%r (case_sensitive=%s)",
        treatment_map["00"],
        treatment_map["10"],
        treatment_map["01"],
        treatment_map["11"],
        bool(treatment_case_sensitive),
    )

    if prep and bool(prep.get("enable", False)):
        LOG.info(
            "• prep: enable=True, mode=%s, target_time=%s, tol=±%s h, align_corners=%s",
            prep.get("mode", "last"),
            prep.get("target_time"),
            prep.get("tolerance", 0.51),
            prep.get("align_corners", False),
        )

        df = prepare_for_logic_symmetry(
            df,
            response_channel=response_channel,
            design_by=design_by,
            batch_col=batch_col,
            treatment_map=treatment_map,
            mode=str(prep.get("mode", "last")),
            target_time=prep.get("target_time"),
            tolerance=float(prep.get("tolerance", 0.51)),
            align_corners=bool(prep.get("align_corners", False)),
            case_sensitive=bool(prep.get("case_sensitive_treatments", treatment_case_sensitive)),
            time_column=str(prep.get("time_column", "time")),
        )
        LOG.info("• prep: rows after selection = %d", len(df))
    else:
        LOG.info("• prep: disabled")

    cfg = MappingConfig(
        treatment_map=treatment_map,
        case_sensitive=bool(treatment_case_sensitive),
        design_by=design_by,
        batch_col=batch_col,
        response_channel=response_channel,
        replicate_stat=replicate_stat,
    )
    points, _per_corner = resolve_and_aggregate(df, cfg)
    LOG.info("• aggregated groups (design×batch) = %d", len(points))

    metrics_rows = []
    for _, r in points.iterrows():
        cs = CornerStats(
            b00=float(r["b00"]),
            b10=float(r["b10"]),
            b01=float(r["b01"]),
            b11=float(r["b11"]),
            n00=int(r["n00"]),
            n10=int(r["n10"]),
            n01=int(r["n01"]),
            n11=int(r["n11"]),
            sd00=float(r["sd00"]),
            sd10=float(r["sd10"]),
            sd01=float(r["sd01"]),
            sd11=float(r["sd11"]),
        )
        met = compute_metrics(cs)
        metrics_rows.append(met)
    met_df = pd.DataFrame.from_records(metrics_rows, index=points.index)

    points_full = pd.concat([points.reset_index(drop=True), met_df.reset_index(drop=True)], axis=1)

    points_full["baseline_corner"] = points_full.apply(_pick_baseline_corner, axis=1)
    points_full["baseline_value"] = points_full[["b00", "b10", "b01", "b11"]].min(axis=1).astype(float)

    if enc_cfg.hue is None or str(enc_cfg.hue).lower() in {"baseline", "baseline_corner", "min", "min_corner"}:
        enc_cfg = EncodingConfig(
            size_by=enc_cfg.size_by,
            size_fixed=enc_cfg.size_fixed,
            hue="baseline_corner",
            alpha_by=enc_cfg.alpha_by,
            alpha_min=enc_cfg.alpha_min,
            alpha_max=enc_cfg.alpha_max,
            shape_by=enc_cfg.shape_by,
            shape_cycle=enc_cfg.shape_cycle,
            shape_max_categories=enc_cfg.shape_max_categories,
        )
        LOG.info("• hue: using baseline_corner (min of {b00,b10,b01,b11})")
    else:
        LOG.info("• hue: using column %r", enc_cfg.hue)

    base_counts = points_full["baseline_corner"].value_counts(dropna=False).to_dict()
    LOG.info("• baseline distribution: %s", base_counts)
    LOG.info("• overlay: enable=%s | gate_set=%s | mode=%s", bool(overlay_enable), overlay_gate_set, overlay_style.mode)

    for col in {enc_cfg.hue, enc_cfg.alpha_by, enc_cfg.shape_by} - {None}:
        if col not in points_full.columns and col in df.columns:
            keys = design_by + [batch_col]
            rep = df.groupby(keys)[col].first().reset_index()
            points_full = points_full.merge(rep, on=keys, how="left")

    encoded = apply_encodings(points_full, enc_cfg)

    title = f"Logic–Symmetry • {response_channel}"
    fig, ax = draw_scatter(
        encoded,
        hue_col=(enc_cfg.hue if enc_cfg.hue else None),
        visuals=vis_cfg,
        uncertainty_mode=uncertainty_mode,
        overlay_cfg=(overlay_style if overlay_enable else None),
        overlay_gate_set=(overlay_gate_set if overlay_enable else None),
        title=title,
        figsize=figsize,
        dpi=dpi,
    )

    csv_cols = (
        design_by
        + [batch_col]
        + [
            "n00",
            "n10",
            "n01",
            "n11",
            "b00",
            "b10",
            "b01",
            "b11",
            "sd00",
            "sd10",
            "sd01",
            "sd11",
            "r",
            "log_r",
            "cv",
            "u00",
            "u10",
            "u01",
            "u11",
            "L",
            "A",
            "baseline_corner",
            "baseline_value",
        ]
    )
    encoded_cols = ["size_value", "hue_value", "alpha_value", "shape_value"]
    csv_out = encoded[csv_cols + encoded_cols].copy()

    # ---- Enforce contract dtypes (assertive) ----
    # minimal, critical enforcement to avoid accidental coercions via I/O
    def _cast_if_present(s: pd.Series, dtype: str) -> pd.Series:
        try:
            return s.astype(dtype)
        except Exception:
            # Let upstream validation surface a precise error if casting fails
            return s

    # Required string columns
    if "baseline_corner" in csv_out.columns:
        csv_out["baseline_corner"] = _cast_if_present(csv_out["baseline_corner"].astype("string"), "string")
    for col in ("shape_value", "hue_value"):
        if col in csv_out.columns:
            csv_out[col] = _cast_if_present(csv_out[col].astype("string"), "string")

    out_dir = Path(output_dir)
    base = f"{base_name}"

    plot_paths = save_plot(fig, out_dir, base, out_formats, dpi)

    # Optional companion CSV for human inspection (off by default to avoid duplication)
    write_csv_flag = bool(_dget(output, "write_csv", False))
    csv_path = None
    if write_csv_flag:
        csv_path = write_csv(csv_out, out_dir, base)

    LOG.info(
        "✓ logic_symmetry wrote: %s.[%s]%s",
        str(out_dir / base),
        ",".join(out_formats),
        (" and " + (out_dir / f"{base}.csv").name) if write_csv_flag else "",
    )

    return LogicSymmetryResult(table=csv_out, fig=fig, plot_paths=plot_paths, csv_path=csv_path)
