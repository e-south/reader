"""Pure logic-symmetry summary and figure operations."""

from __future__ import annotations

import logging

import pandas as pd

from .encodings import EncodingConfig, apply_encodings
from .extract_corners import MappingConfig, resolve_and_aggregate
from .metrics import CornerStats, compute_metrics
from .overlay import OverlayStyle
from .prep import prepare_for_logic_symmetry
from .render import VisualConfig, draw_scatter

LOG = logging.getLogger(__name__)


def _dget(values: dict | None, key: str, default):
    if values is None:
        return default
    if key in values and values[key] is not None:
        return values[key]
    return default


def _pick_baseline_corner(row: pd.Series) -> str:
    corners = ("b00", "b10", "b01", "b11")
    values = {corner: float(row[corner]) for corner in corners}
    minimum = min(values.values())
    for corner in corners:
        if values[corner] == minimum:
            return corner.removeprefix("b")
    return "00"


def summarize_logic_symmetry(
    df: pd.DataFrame,
    *,
    response_channel: str,
    design_by: list[str] | None = None,
    batch_col: str = "batch",
    treatment_column: str | None = None,
    treatment_map: dict[str, str] | None = None,
    treatment_case_sensitive: bool = True,
    observation_stat: str = "mean",
    prep: dict | None = None,
) -> pd.DataFrame:
    """Compute one contract-ready logic-symmetry row per design and batch."""

    design_by = list(design_by or ["design_id"])
    if treatment_map is None or set(treatment_map) != {"00", "10", "01", "11"}:
        raise ValueError(
            "treatment_map must be provided with keys {'00','10','01','11'} and single exact labels as values"
        )
    if observation_stat not in {"mean", "median"}:
        raise ValueError(f"observation_stat must be 'mean' or 'median', got {observation_stat!r}")

    LOG.info("logic_symmetry: computing summary")
    LOG.info("response_channel=%s | design_by=%s | batch_col=%s", response_channel, design_by, batch_col)
    LOG.info(
        "state_map: 00=%r | 10=%r | 01=%r | 11=%r | case_sensitive=%s",
        treatment_map["00"],
        treatment_map["10"],
        treatment_map["01"],
        treatment_map["11"],
        treatment_case_sensitive,
    )

    if prep and bool(prep.get("enable", False)):
        df = prepare_for_logic_symmetry(
            df,
            response_channel=response_channel,
            design_by=design_by,
            batch_col=batch_col,
            treatment_map=treatment_map,
            treatment_column=treatment_column,
            mode=str(prep.get("mode", "last")),
            target_time=prep.get("target_time"),
            tolerance=float(prep.get("tolerance", 0.51)),
            align_corners=bool(prep.get("align_corners", False)),
            case_sensitive=bool(prep.get("case_sensitive_treatments", treatment_case_sensitive)),
            time_column=str(prep.get("time_column", "time")),
        )
        LOG.info("logic_symmetry: rows after time selection=%d", len(df))

    mapping = MappingConfig(
        treatment_map=treatment_map,
        case_sensitive=treatment_case_sensitive,
        treatment_column=treatment_column,
        design_by=design_by,
        batch_col=batch_col,
        response_channel=response_channel,
        observation_stat=observation_stat,
    )
    points, _ = resolve_and_aggregate(df, mapping)
    LOG.info("logic_symmetry: aggregated groups=%d", len(points))

    metrics = []
    for _, row in points.iterrows():
        metrics.append(
            compute_metrics(
                CornerStats(
                    b00=float(row["b00"]),
                    b10=float(row["b10"]),
                    b01=float(row["b01"]),
                    b11=float(row["b11"]),
                    n00=int(row["n00"]),
                    n10=int(row["n10"]),
                    n01=int(row["n01"]),
                    n11=int(row["n11"]),
                    sd00=float(row["sd00"]),
                    sd10=float(row["sd10"]),
                    sd01=float(row["sd01"]),
                    sd11=float(row["sd11"]),
                )
            )
        )
    metric_table = pd.DataFrame.from_records(metrics, index=points.index)
    summary = pd.concat([points.reset_index(drop=True), metric_table.reset_index(drop=True)], axis=1)
    summary["baseline_corner"] = summary.apply(_pick_baseline_corner, axis=1).astype("string")
    summary["baseline_value"] = summary[["b00", "b10", "b01", "b11"]].min(axis=1).astype(float)

    value_columns = [
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
    identity_columns = list(dict.fromkeys([*design_by, batch_col]))
    return summary[[*identity_columns, *value_columns]].copy()


def render_logic_symmetry(
    table: pd.DataFrame,
    *,
    title: str = "Logic symmetry",
    dispersion: str = "halo",
    encodings: dict | None = None,
    ideals_overlay: dict | None = None,
    visuals: dict | None = None,
    figsize: tuple[float, float] = (7, 6),
    dpi: int = 300,
):
    """Render a logic-symmetry figure from a persisted summary table."""

    if dispersion not in {"none", "bars", "halo"}:
        raise ValueError(f"dispersion must be one of 'none'|'bars'|'halo', got {dispersion!r}")

    hue = _dget(encodings, "hue", None)
    if hue is None or str(hue).lower() in {"baseline", "baseline_corner", "min", "min_corner"}:
        hue = "baseline_corner"
    encoding = EncodingConfig(
        size_by=str(_dget(encodings, "size_by", "log_r")),
        size_fixed=float(_dget(encodings, "size_fixed", 80.0)),
        hue=str(hue),
        alpha_by=_dget(encodings, "alpha_by", "batch"),
        alpha_min=float(_dget(encodings, "alpha_min", 0.35)),
        alpha_max=float(_dget(encodings, "alpha_max", 1.0)),
        shape_by=_dget(encodings, "shape_by", None),
        shape_cycle=list(_dget(encodings, "shape_cycle", ["o", "s", "^", "D", "P", "X", "v", "*"])),
        shape_max_categories=_dget(encodings, "shape_max_categories", None),
    )
    for column in {encoding.hue, encoding.alpha_by, encoding.shape_by} - {None}:
        if column not in table.columns:
            raise ValueError(f"Logic-symmetry encoding refers to missing column {column!r}")
    encoded = apply_encodings(table, encoding)

    overlay_enabled = bool(_dget(ideals_overlay, "enable", False))
    overlay_gate_set = str(_dget(ideals_overlay, "gate_set", "logic_family"))
    overlay_values = _dget(ideals_overlay, "style", {})
    overlay_mode = _dget(overlay_values, "mode", None)
    if overlay_mode is None:
        overlay_mode = "tiles" if overlay_gate_set.startswith("tiles") else "dot"
    overlay = OverlayStyle(
        mode=str(overlay_mode),
        alpha=float(_dget(overlay_values, "alpha", 0.25)),
        size=float(_dget(overlay_values, "size", 40.0)),
        face_color=str(_dget(overlay_values, "face_color", "#FFFFFF")),
        edge_color=str(_dget(overlay_values, "edge_color", _dget(overlay_values, "color", "#888888"))),
        show_labels=bool(_dget(overlay_values, "show_labels", True)),
        label_offset=float(_dget(overlay_values, "label_offset", 0.02)),
        label_line_height=float(_dget(overlay_values, "label_line_height", 0.018)),
        label_fontsize=int(_dget(overlay_values, "label_fontsize", 12)),
        tile_cell_w=float(_dget(overlay_values, "tile_cell_w", 0.035)),
        tile_cell_h=float(_dget(overlay_values, "tile_cell_h", 0.035)),
        tile_gap=float(_dget(overlay_values, "tile_gap", 0.0)),
        tile_edge_width=float(_dget(overlay_values, "tile_edge_width", 0.6)),
        tiles_stack_multiple=bool(_dget(overlay_values, "tiles_stack_multiple", True)),
    )

    visual = VisualConfig(
        xlim=tuple(_dget(visuals, "xlim", (-1.02, 1.02))),
        ylim=tuple(_dget(visuals, "ylim", (-1.02, 1.02))),
        grid=bool(_dget(visuals, "grid", True)),
        color=str(_dget(visuals, "color", "#6e6e6e")),
        annotate_designs=bool(_dget(visuals, "annotate_designs", False)),
        design_label_col=_dget(visuals, "design_label_col", None),
        label_fontsize=int(_dget(visuals, "label_fontsize", 12)),
        label_offset=float(_dget(visuals, "label_offset", 0.02)),
        axis_label_fontsize=int(_dget(visuals, "axis_label_fontsize", 16)),
        tick_label_fontsize=int(_dget(visuals, "tick_label_fontsize", 14)),
        title_fontsize=int(_dget(visuals, "title_fontsize", 18)),
        legend_fontsize=int(_dget(visuals, "legend_fontsize", 12)),
    )
    figure, _ = draw_scatter(
        encoded,
        hue_col=encoding.hue,
        visuals=visual,
        dispersion_mode=dispersion,
        overlay_cfg=overlay if overlay_enabled else None,
        overlay_gate_set=overlay_gate_set if overlay_enabled else None,
        title=title,
        figsize=figsize,
        dpi=dpi,
    )
    return figure
