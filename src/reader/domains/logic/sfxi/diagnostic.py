"""Pure preparation and rendering for persisted SFXI diagnostics.

The diagnostic joins an annotated measurement trace to the already-persisted
vec8 summary. It does not select a new acquisition time or recompute vec8.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from reader.domains.plate_reader.plots.dual_reporter_triptych import (
    DualReporterTriptychData,
    build_triptych_data,
)

STATE_ORDER = ("00", "10", "01", "11")
_LOGIC_COLUMNS = {state: f"v{state}" for state in STATE_ORDER}
_INTENSITY_COLUMNS = {state: f"y{state}_star" for state in STATE_ORDER}
_STATE_COLORS = {
    "00": "#364152",
    "10": "#237A70",
    "01": "#2F65D9",
    "11": "#B82E3F",
}
_STATE_MARKERS = {"00": "o", "10": "s", "01": "^", "11": "D"}


@dataclass(frozen=True)
class SFXIDiagnosticData:
    design_id: str
    selected_time_h: float
    reference_design_id: str
    time_column: str
    state_column: str
    triptych: DualReporterTriptychData
    logic_components: dict[str, float]
    intensity_components: dict[str, float]


def prepare_sfxi_diagnostics(
    annotated: pd.DataFrame,
    vec8: pd.DataFrame,
    *,
    treatment_column: str,
    treatment_map: Mapping[str, str],
    treatment_case_sensitive: bool,
    time_column: str,
    growth_channel: str,
    response_channel: str,
    design_ids: Sequence[str] | None = None,
    time_atol: float = 1e-9,
    trajectory_ci: float = 95.0,
    trajectory_bootstraps: int = 300,
) -> tuple[SFXIDiagnosticData, ...]:
    """Prepare one diagnostic per persisted vec8 design.

    Vec8 row order is the default output order. An explicit ``design_ids``
    selection preserves caller order and fails when any id is unavailable.
    """

    _require_columns(
        annotated,
        ("design_id", treatment_column, time_column, "channel", "value", "position"),
        where="SFXI diagnostic annotated data",
    )
    _require_columns(
        vec8,
        (
            "design_id",
            "time_selected_h",
            "reference_design_id",
            *_LOGIC_COLUMNS.values(),
            *_INTENSITY_COLUMNS.values(),
        ),
        where="SFXI diagnostic vec8 data",
    )
    if annotated.empty:
        raise ValueError("SFXI diagnostic annotated data must not be empty")
    if vec8.empty:
        raise ValueError("SFXI diagnostic vec8 data must not be empty")
    if not math.isfinite(float(time_atol)) or float(time_atol) < 0.0:
        raise ValueError("SFXI diagnostic time_atol must be finite and non-negative")

    source_to_state = _source_to_state(
        treatment_map,
        case_sensitive=treatment_case_sensitive,
    )
    vec8_work = vec8.copy().reset_index(drop=True)
    vec8_work["design_id"] = _non_empty_text_series(vec8_work["design_id"], field="vec8.design_id")
    duplicate_designs = vec8_work.loc[vec8_work["design_id"].duplicated(keep=False), "design_id"].unique().tolist()
    if duplicate_designs:
        raise ValueError(f"SFXI diagnostic vec8 has duplicate design_id values: {duplicate_designs}")

    selected_designs = _selected_design_ids(vec8_work, design_ids=design_ids)
    annotated_work = annotated.copy()
    annotated_work["design_id"] = _non_empty_text_series(
        annotated_work["design_id"],
        field="annotated.design_id",
    )
    source_values = annotated_work[treatment_column].astype(str)
    if not treatment_case_sensitive:
        source_values = source_values.str.strip().str.casefold()
    annotated_work["__sfxi_state"] = source_values.map(source_to_state)
    annotated_work = annotated_work[annotated_work["__sfxi_state"].isin(STATE_ORDER)].copy()

    prepared: list[SFXIDiagnosticData] = []
    by_design = vec8_work.set_index("design_id", drop=False)
    for design_id in selected_designs:
        vec8_row = by_design.loc[design_id]
        selected_time_h = _finite_scalar(vec8_row["time_selected_h"], field=f"{design_id}.time_selected_h")
        reference_design_id = _non_empty_text(
            vec8_row["reference_design_id"],
            field=f"{design_id}.reference_design_id",
        )
        design_traces = annotated_work[annotated_work["design_id"].eq(design_id)].copy()
        if design_traces.empty:
            raise ValueError(f"SFXI diagnostic has no annotated rows for vec8 design_id {design_id!r}")
        _require_state_channel_coverage(
            design_traces,
            design_id=design_id,
            channels=(growth_channel, response_channel),
        )
        _require_persisted_time_in_traces(
            design_traces,
            design_id=design_id,
            selected_time_h=selected_time_h,
            time_column=time_column,
            channels=(growth_channel, response_channel),
            time_atol=float(time_atol),
        )
        triptych = build_triptych_data(
            design_traces,
            time_col=time_column,
            treatment_col="__sfxi_state",
            growth_channel=growth_channel,
            ratio_channel=response_channel,
            snapshot_channel=response_channel,
            snapshot_time=selected_time_h,
            treatment_order=STATE_ORDER,
            time_atol=float(time_atol),
            trajectory_ci=float(trajectory_ci),
            trajectory_bootstraps=int(trajectory_bootstraps),
        )
        logic_components = {
            state: _finite_scalar(vec8_row[column], field=f"{design_id}.{column}")
            for state, column in _LOGIC_COLUMNS.items()
        }
        invalid_logic = {state: value for state, value in logic_components.items() if value < 0.0 or value > 1.0}
        if invalid_logic:
            raise ValueError(
                f"SFXI diagnostic logic-shape components must lie in [0, 1] for {design_id!r}: {invalid_logic}"
            )
        intensity_components = {
            state: _finite_scalar(vec8_row[column], field=f"{design_id}.{column}")
            for state, column in _INTENSITY_COLUMNS.items()
        }
        prepared.append(
            SFXIDiagnosticData(
                design_id=design_id,
                selected_time_h=selected_time_h,
                reference_design_id=reference_design_id,
                time_column=time_column,
                state_column="__sfxi_state",
                triptych=triptych,
                logic_components=logic_components,
                intensity_components=intensity_components,
            )
        )
    return tuple(prepared)


def render_sfxi_diagnostic(
    data: SFXIDiagnosticData,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (15.0, 5.5),
    dpi: int = 300,
) -> Any:
    """Render trajectories beside the persisted vec8 components."""

    if len(figsize) != 2 or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in figsize):
        raise ValueError("SFXI diagnostic figsize must contain two positive finite values")
    if int(dpi) < 1:
        raise ValueError("SFXI diagnostic dpi must be positive")

    from reader.domains.plate_reader.plots.dual_reporter_triptych_render import (  # noqa: PLC0415
        render_dual_reporter_triptych,
    )

    figure = render_dual_reporter_triptych(
        data.triptych,
        time_col=data.time_column,
        treatment_col=data.state_column,
        title=None,
        figsize=figsize,
    )
    figure.set_dpi(int(dpi))
    figure.axes[0].set_title("Growth trajectory")
    figure.axes[1].set_title("Response trajectory")
    snapshot_spec = figure.axes[2].get_subplotspec()
    figure.delaxes(figure.axes[2])
    component_grid = snapshot_spec.subgridspec(2, 1, hspace=0.7)
    logic_axis = figure.add_subplot(component_grid[0, 0])
    intensity_axis = figure.add_subplot(component_grid[1, 0])
    _render_components(
        logic_axis,
        data.logic_components,
        title="Logic shape",
        x_label="Persisted value (unit interval)",
        x_limits=(-0.05, 1.05),
    )
    _render_components(
        intensity_axis,
        data.intensity_components,
        title="Relative intensity",
        x_label="Persisted value (log2)",
        draw_zero=True,
    )

    heading = title.strip() if isinstance(title, str) and title.strip() else "SFXI diagnostic"
    figure.suptitle(
        f"{heading} · {data.design_id}\n"
        f"selected time {data.selected_time_h:g} h · reference {data.reference_design_id}",
        fontsize=13,
    )
    return figure


def _render_components(
    axis: Any,
    values: Mapping[str, float],
    *,
    title: str,
    x_label: str,
    x_limits: tuple[float, float] | None = None,
    draw_zero: bool = False,
) -> None:
    positions = np.arange(len(STATE_ORDER), dtype=float)
    component_values = np.array([values[state] for state in STATE_ORDER], dtype=float)
    if draw_zero:
        axis.axvline(0.0, color="#8A94A6", linewidth=0.9)
    for position, state, value in zip(positions, STATE_ORDER, component_values, strict=True):
        axis.hlines(position, min(0.0, value), max(0.0, value), color=_STATE_COLORS[state], linewidth=1.5)
        axis.scatter(
            value,
            position,
            color=_STATE_COLORS[state],
            marker=_STATE_MARKERS[state],
            s=34,
            zorder=3,
        )
    axis.set_yticks(positions, labels=STATE_ORDER)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("State")
    if x_limits is not None:
        axis.set_xlim(*x_limits)
    axis.grid(axis="x", color="#D8DEE8", linewidth=0.6, alpha=0.7)


def _selected_design_ids(vec8: pd.DataFrame, *, design_ids: Sequence[str] | None) -> tuple[str, ...]:
    available = tuple(vec8["design_id"].astype(str).tolist())
    if design_ids is None:
        return available
    selected = tuple(_non_empty_text(value, field="design_ids") for value in design_ids)
    if not selected:
        raise ValueError("SFXI diagnostic design_ids must not be empty when provided")
    if len(set(selected)) != len(selected):
        raise ValueError("SFXI diagnostic design_ids must be unique")
    missing = [design_id for design_id in selected if design_id not in set(available)]
    if missing:
        raise ValueError(f"SFXI diagnostic design_ids are absent from vec8: {missing}")
    return selected


def _source_to_state(treatment_map: Mapping[str, str], *, case_sensitive: bool) -> dict[str, str]:
    if set(treatment_map) != set(STATE_ORDER):
        raise ValueError("SFXI diagnostic treatment_map must contain exactly 00, 10, 01, and 11")
    reverse: dict[str, str] = {}
    for state in STATE_ORDER:
        source_value = _non_empty_text(treatment_map[state], field=f"treatment_map.{state}")
        key = source_value if case_sensitive else source_value.strip().casefold()
        if key in reverse:
            raise ValueError("SFXI diagnostic treatment_map source values must be unique")
        reverse[key] = state
    return reverse


def _require_state_channel_coverage(frame: pd.DataFrame, *, design_id: str, channels: Sequence[str]) -> None:
    for channel in channels:
        observed = set(frame.loc[frame["channel"].astype(str).eq(str(channel)), "__sfxi_state"].astype(str))
        missing = [state for state in STATE_ORDER if state not in observed]
        if missing:
            raise ValueError(
                f"SFXI diagnostic design {design_id!r} channel {channel!r} is missing state traces: {missing}"
            )


def _require_persisted_time_in_traces(
    frame: pd.DataFrame,
    *,
    design_id: str,
    selected_time_h: float,
    time_column: str,
    channels: Sequence[str],
    time_atol: float,
) -> None:
    times = pd.to_numeric(frame[time_column], errors="coerce")
    for channel in channels:
        channel_times = times[frame["channel"].astype(str).eq(str(channel))].dropna().to_numpy(dtype=float)
        if not np.isclose(channel_times, selected_time_h, rtol=0.0, atol=time_atol).any():
            raise ValueError(
                f"SFXI diagnostic persisted selection time {selected_time_h:g} h for {design_id!r} "
                f"is absent from channel {channel!r} traces"
            )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, where: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{where} is missing columns: {missing}")


def _finite_scalar(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"SFXI diagnostic {field} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"SFXI diagnostic {field} must be a finite number")
    return number


def _non_empty_text_series(series: pd.Series, *, field: str) -> pd.Series:
    values = series.astype(str).str.strip()
    if values.eq("").any() or values.str.casefold().eq("nan").any():
        raise ValueError(f"SFXI diagnostic {field} must contain non-empty values")
    return values


def _non_empty_text(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "<na>"}:
        raise ValueError(f"SFXI diagnostic {field} must be a non-empty string")
    return text


__all__ = [
    "SFXIDiagnosticData",
    "prepare_sfxi_diagnostics",
    "render_sfxi_diagnostic",
]
