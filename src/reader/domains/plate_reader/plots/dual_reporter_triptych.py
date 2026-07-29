"""Domain-owned preparation and Altair rendering for dual-reporter triptychs.

The capability accepts explicit tidy data and has no workbench, record-store,
or notebook-lifecycle dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from reader.domains.plate_reader.plots.common import bootstrap_mean_interval

DEFAULT_TRIPTYCH_PANEL_SIZE = 260
DEFAULT_TRIPTYCH_SPACING = 16
DEFAULT_TRAJECTORY_CI = 95.0
DEFAULT_TRAJECTORY_BOOTSTRAPS = 300


@dataclass(frozen=True)
class DualReporterTriptychData:
    od600_time: pd.DataFrame
    ratio_time: pd.DataFrame
    snapshot_stats: pd.DataFrame
    snapshot_points: pd.DataFrame
    treatment_order: tuple[str, ...]
    missing_treatments: tuple[str, ...]
    snapshot_time: float
    growth_channel: str
    ratio_channel: str
    snapshot_channel: str
    trajectory_ci: float


def build_triptych_data(
    df: pd.DataFrame,
    *,
    time_col: str,
    treatment_col: str,
    growth_channel: str = "OD600",
    ratio_channel: str = "YFP/CFP",
    snapshot_channel: str | None = None,
    snapshot_time: float,
    treatment_order: list[str] | tuple[str, ...] | None = None,
    time_atol: float = 1e-9,
    trajectory_ci: float = DEFAULT_TRAJECTORY_CI,
    trajectory_bootstraps: int = DEFAULT_TRAJECTORY_BOOTSTRAPS,
) -> DualReporterTriptychData:
    """Prepare the neutral dual-reporter triptych data contract.

    This is intentionally independent of SFXI semantics. It only requires a
    tidy dual-reporter dataframe with channels for growth, ratio kinetics, and
    a snapshot channel.
    """
    snapshot_channel = snapshot_channel or ratio_channel
    if not 0.0 < float(trajectory_ci) < 100.0:
        raise ValueError("dual_reporter_triptych: trajectory_ci must lie strictly between 0 and 100")
    if int(trajectory_bootstraps) < 1:
        raise ValueError("dual_reporter_triptych: trajectory_bootstraps must be positive")
    required_channels = [growth_channel, ratio_channel, snapshot_channel]
    _require_columns(df, ["channel", "value", time_col, treatment_col], where="dual_reporter_triptych")

    work = df.copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=[time_col, "value", treatment_col, "channel"])
    if work.empty:
        raise ValueError("dual_reporter_triptych: no usable rows after numeric/time cleanup")
    work[treatment_col] = work[treatment_col].astype(str)
    work["channel"] = work["channel"].astype(str)

    order = _resolve_treatment_order(work, treatment_col=treatment_col, treatment_order=treatment_order)
    if treatment_order is not None:
        work = work[work[treatment_col].isin(order)].copy()
        if work.empty:
            raise ValueError("dual_reporter_triptych: no rows match the requested treatment order")
    _require_channels(work, required_channels)
    observed = set(work[treatment_col].dropna().astype(str).unique().tolist())
    missing_treatments = tuple(value for value in order if value not in observed)

    summary_options = {
        "time_col": time_col,
        "treatment_col": treatment_col,
        "order": order,
        "ci": float(trajectory_ci),
        "ci_boot": int(trajectory_bootstraps),
    }
    od600_time = _summarize_time(work, channel=growth_channel, **summary_options)
    ratio_time = _summarize_time(work, channel=ratio_channel, **summary_options)
    snapshot_stats, snapshot_points = _summarize_snapshot(
        work,
        channel=snapshot_channel,
        time_col=time_col,
        treatment_col=treatment_col,
        snapshot_time=float(snapshot_time),
        order=order,
        time_atol=float(time_atol),
    )
    return DualReporterTriptychData(
        od600_time=od600_time,
        ratio_time=ratio_time,
        snapshot_stats=snapshot_stats,
        snapshot_points=snapshot_points,
        treatment_order=tuple(order),
        missing_treatments=missing_treatments,
        snapshot_time=float(snapshot_time),
        growth_channel=str(growth_channel),
        ratio_channel=str(ratio_channel),
        snapshot_channel=str(snapshot_channel),
        trajectory_ci=float(trajectory_ci),
    )


def build_dual_reporter_triptych_chart(
    *,
    alt: Any,
    pd_module: Any,
    data: DualReporterTriptychData,
    time_col: str,
    treatment_col: str,
    acquisition_transition_time_h: float | None = None,
    width: int = DEFAULT_TRIPTYCH_PANEL_SIZE,
    height: int | None = None,
    spacing: int = DEFAULT_TRIPTYCH_SPACING,
    treatment_title: str | None = None,
) -> Any:
    if data.od600_time.empty:
        raise ValueError("dual_reporter_triptych: no OD600 time-series data available")
    if data.ratio_time.empty:
        raise ValueError("dual_reporter_triptych: no ratio time-series data available")
    if data.snapshot_stats.empty:
        raise ValueError("dual_reporter_triptych: no snapshot data available")

    order = list(data.treatment_order)
    panel_height = int(height if height is not None else width)
    color = alt.Color(
        f"{treatment_col}:N",
        sort=order,
        scale=alt.Scale(domain=order),
        legend=alt.Legend(orient="bottom", title=treatment_title or _display_column_title(treatment_col)),
    )

    od600_chart = _time_chart(
        alt=alt,
        pd_module=pd_module,
        frame=data.od600_time,
        time_col=time_col,
        treatment_col=treatment_col,
        y_title=data.growth_channel,
        color=color,
        order=order,
        treatment_title=treatment_title or _display_column_title(treatment_col),
        snapshot_time=data.snapshot_time,
        acquisition_transition_time_h=acquisition_transition_time_h,
        width=width,
        height=panel_height,
    )
    ratio_chart = _time_chart(
        alt=alt,
        pd_module=pd_module,
        frame=data.ratio_time,
        time_col=time_col,
        treatment_col=treatment_col,
        y_title=data.ratio_channel,
        color=color,
        order=order,
        treatment_title=treatment_title or _display_column_title(treatment_col),
        snapshot_time=data.snapshot_time,
        acquisition_transition_time_h=acquisition_transition_time_h,
        width=width,
        height=panel_height,
    )
    snapshot_chart = _snapshot_chart(
        alt=alt,
        frame=data.snapshot_stats,
        points=data.snapshot_points,
        treatment_col=treatment_col,
        y_title=f"{data.snapshot_channel} snapshot",
        order=order,
        treatment_title=treatment_title or _display_column_title(treatment_col),
        width=width,
        height=panel_height,
    )
    return (
        alt.hconcat(od600_chart, ratio_chart, snapshot_chart, spacing=spacing)
        .resolve_scale(color="shared", strokeDash="shared")
        .configure(background="white")
        .configure_view(fill="white")
        .configure_axis(
            domain=True,
            domainColor="black",
            domainWidth=1,
            tickColor="black",
            labelColor="black",
            titleColor="black",
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_legend(labelColor="black", titleColor="black", labelFontSize=12, titleFontSize=12)
        .configure_title(color="black", fontSize=14)
        .configure_text(color="black", fontSize=12)
    )


def summarize_design_context(
    df: pd.DataFrame,
    *,
    primary_col: str,
    primary_value: object,
    preferred_columns: tuple[str, ...] = ("design_id_alias", "design_id", "id", "sequence", "strain", "medium"),
    max_value_chars: int = 84,
) -> list[tuple[str, str]]:
    """Return compact identity rows for a selected design/genotype."""

    rows = [(str(primary_col), _compact_context_value(primary_value, max_chars=max_value_chars))]
    seen = {str(primary_col)}
    for column in preferred_columns:
        column = str(column)
        if column in seen or column not in df.columns:
            continue
        values = _unique_context_values(df[column])
        if not values:
            continue
        if len(values) == 1:
            display = _compact_context_value(values[0], max_chars=max_value_chars)
        else:
            display = f"{len(values)} values: {_compact_context_value(values[0], max_chars=max_value_chars)}"
        rows.append((column, display))
        seen.add(column)
    return rows


def choose_time(times: list[float] | tuple[float, ...], target: float | None, mode: str) -> float | None:
    if not times:
        return None
    time_list = sorted(float(t) for t in times)
    if target is None:
        return time_list[-1]
    target = float(target)
    if mode == "exact":
        for time_value in time_list:
            if abs(time_value - target) <= 1e-12:
                return time_value
        return None
    if mode == "nearest":
        return min(time_list, key=lambda time_value: abs(time_value - target))
    if mode == "last_before":
        candidates = [time_value for time_value in time_list if time_value <= target]
        return max(candidates) if candidates else None
    if mode == "first_after":
        candidates = [time_value for time_value in time_list if time_value >= target]
        return min(candidates) if candidates else None
    raise ValueError("dual_reporter_triptych: time mode must be nearest, last_before, first_after, or exact")


def _unique_context_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.dropna().tolist():
        text = str(value).strip()
        if not text or text.casefold() == "nan" or text in seen:
            continue
        values.append(text)
        seen.add(text)
    return values


def _compact_context_value(value: object, *, max_chars: int) -> str:
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    side = max(8, (max_chars - 3) // 2)
    return f"{text[:side]}...{text[-side:]}"


def _require_columns(df: pd.DataFrame, columns: list[str], *, where: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing column(s): {', '.join(missing)}")


def _require_channels(df: pd.DataFrame, channels: list[str]) -> None:
    available = {str(value) for value in df["channel"].dropna().unique().tolist()}
    missing = [channel for channel in channels if str(channel) not in available]
    if missing:
        options = ", ".join(sorted(available))
        raise ValueError(f"dual_reporter_triptych: requested channel(s) not found: {missing}. Available: {options}")


def _resolve_treatment_order(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    treatment_order: list[str] | tuple[str, ...] | None,
) -> list[str]:
    observed = [str(value) for value in df[treatment_col].dropna().unique().tolist()]
    if treatment_order is not None:
        order = []
        seen = set()
        for value in treatment_order:
            item = str(value)
            if item not in seen:
                order.append(item)
                seen.add(item)
        if not order:
            raise ValueError("dual_reporter_triptych: treatment_order must not be empty when provided")
        return order
    return sorted(observed)


def _sort_by_treatment(
    df: pd.DataFrame, *, treatment_col: str, order: list[str], extra_sort: list[str]
) -> pd.DataFrame:
    if df.empty:
        return df
    order_map = {value: idx for idx, value in enumerate(order)}
    sorted_df = df.copy()
    sorted_df["__treatment_order"] = sorted_df[treatment_col].astype(str).map(order_map).fillna(len(order_map))
    sorted_df = sorted_df.sort_values(["__treatment_order", *extra_sort]).drop(columns=["__treatment_order"])
    return sorted_df.reset_index(drop=True)


def _summarize_time(
    df: pd.DataFrame,
    *,
    channel: str,
    time_col: str,
    treatment_col: str,
    order: list[str],
    ci: float,
    ci_boot: int,
) -> pd.DataFrame:
    work = df[df["channel"].astype(str) == str(channel)].copy()
    if work.empty:
        return pd.DataFrame(columns=[time_col, treatment_col, "y_mean", "y_sd", "y_n", "y_lo", "y_hi"])
    rng = np.random.default_rng(0)
    records: list[dict[str, object]] = []
    for (time_value, treatment), series in work.groupby([time_col, treatment_col], dropna=False, sort=True)["value"]:
        values = series.to_numpy(dtype=float, copy=False)
        mean, lower, upper = bootstrap_mean_interval(values, ci=ci, ci_boot=ci_boot, rng=rng)
        records.append(
            {
                time_col: time_value,
                treatment_col: treatment,
                "y_mean": mean,
                "y_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "y_n": len(values),
                "y_lo": lower,
                "y_hi": upper,
            }
        )
    stats = pd.DataFrame.from_records(records)
    return _sort_by_treatment(stats, treatment_col=treatment_col, order=order, extra_sort=[time_col])


def _summarize_snapshot(
    df: pd.DataFrame,
    *,
    channel: str,
    time_col: str,
    treatment_col: str,
    snapshot_time: float,
    order: list[str],
    time_atol: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[df["channel"].astype(str) == str(channel)].copy()
    if work.empty:
        empty_stats = pd.DataFrame(columns=[treatment_col, "y_mean", "y_sd", "y_n", "y_lo", "y_hi"])
        empty_points = pd.DataFrame(columns=[treatment_col, "value", "replicate_index"])
        return empty_stats, empty_points
    mask = (work[time_col] - float(snapshot_time)).abs() <= float(time_atol)
    snapped = work[mask].copy()
    if snapped.empty:
        empty_stats = pd.DataFrame(columns=[treatment_col, "y_mean", "y_sd", "y_n", "y_lo", "y_hi"])
        empty_points = pd.DataFrame(columns=[treatment_col, "value", "replicate_index"])
        return empty_stats, empty_points
    stats = snapped.groupby(treatment_col, dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
    stats = stats.rename(columns={"mean": "y_mean", "std": "y_sd", "count": "y_n"})
    stats["y_sd"] = stats["y_sd"].fillna(0.0)
    stats["y_lo"] = stats["y_mean"] - stats["y_sd"]
    stats["y_hi"] = stats["y_mean"] + stats["y_sd"]
    stats = _sort_by_treatment(stats, treatment_col=treatment_col, order=order, extra_sort=[])
    snapped["__source_order"] = np.arange(len(snapped), dtype=int)
    stable_identity = [column for column in ("position", "well", "source_name", "sheet_name") if column in snapped]
    point_columns = [treatment_col, "value", *stable_identity, "__source_order"]
    points = _sort_by_treatment(
        snapped[point_columns].copy(),
        treatment_col=treatment_col,
        order=order,
        extra_sort=[*stable_identity, "__source_order"],
    )
    points["replicate_index"] = points.groupby(treatment_col, sort=False).cumcount()
    points = points.drop(columns="__source_order")
    return stats, points


def _time_chart(
    *,
    alt: Any,
    pd_module: Any,
    frame: pd.DataFrame,
    time_col: str,
    treatment_col: str,
    y_title: str,
    color: Any,
    order: list[str],
    treatment_title: str,
    snapshot_time: float,
    acquisition_transition_time_h: float | None,
    width: int,
    height: int,
) -> Any:
    dash_patterns = ([1, 0], [7, 2], [2, 2], [8, 2, 2, 2])
    dash_range = [dash_patterns[index % len(dash_patterns)] for index in range(len(order))]
    base = alt.Chart(frame).encode(
        x=alt.X(f"{time_col}:Q", title="Time (h)", axis=alt.Axis(labelOverlap=False)),
        color=color,
    )
    band = base.mark_area(opacity=0.2).encode(
        y=alt.Y("y_lo:Q", title=y_title),
        y2=alt.Y2("y_hi:Q"),
        tooltip=_time_tooltips(alt, time_col=time_col, treatment_col=treatment_col),
    )
    line = base.mark_line().encode(
        y=alt.Y("y_mean:Q", title=y_title),
        strokeDash=alt.StrokeDash(
            f"{treatment_col}:N",
            sort=order,
            scale=alt.Scale(
                domain=order,
                range=dash_range,
            ),
            legend=alt.Legend(orient="bottom", title=treatment_title),
        ),
        tooltip=_time_tooltips(alt, time_col=time_col, treatment_col=treatment_col),
    )
    layers = [band, line]
    y_max = frame["y_hi"].max()
    if pd_module.isna(y_max):
        y_max = frame["y_mean"].max()
    if pd_module.isna(y_max):
        y_max = 0.0
    rule_df = pd_module.DataFrame({time_col: [float(snapshot_time)], "y": [float(y_max)]})
    layers.append(alt.Chart(rule_df).mark_rule(color="black").encode(x=alt.X(f"{time_col}:Q")))

    if acquisition_transition_time_h is not None:
        try:
            transition_time = float(acquisition_transition_time_h)
        except (TypeError, ValueError):
            transition_time = None
        if transition_time is not None and not pd_module.isna(transition_time):
            transition_df = pd_module.DataFrame({time_col: [transition_time]})
            layers.append(
                alt.Chart(transition_df).mark_rule(color="#5F5F5F", strokeDash=[6, 4]).encode(x=alt.X(f"{time_col}:Q"))
            )

    return alt.layer(*layers).properties(width=width, height=height, title=y_title)


def _snapshot_chart(
    *,
    alt: Any,
    frame: pd.DataFrame,
    points: pd.DataFrame,
    treatment_col: str,
    y_title: str,
    order: list[str],
    treatment_title: str,
    width: int,
    height: int,
) -> Any:
    axis = alt.Axis(labelLimit=0, labelOverlap=False, labelAngle=0, title=treatment_title)
    base = alt.Chart(frame).encode(
        x=alt.X(f"{treatment_col}:N", sort=order, scale=alt.Scale(domain=order), axis=axis),
        y=alt.Y("y_mean:Q", title=y_title),
        tooltip=[
            alt.Tooltip(f"{treatment_col}:N", title="Treatment"),
            alt.Tooltip("y_mean:Q", title="Mean"),
            alt.Tooltip("y_sd:Q", title="SD"),
            alt.Tooltip("y_n:Q", title="N"),
        ],
    )
    layers = [
        base.mark_rule(color="black").encode(y=alt.Y("y_lo:Q"), y2=alt.Y2("y_hi:Q")),
        base.mark_tick(color="black", orient="horizontal", size=8, thickness=1.5).encode(y=alt.Y("y_lo:Q")),
        base.mark_tick(color="black", orient="horizontal", size=8, thickness=1.5).encode(y=alt.Y("y_hi:Q")),
        base.mark_tick(color="#334155", orient="horizontal", size=24, thickness=2.5),
    ]
    if not points.empty:
        layers.append(
            alt.Chart(points)
            .mark_point(filled=True, fill="white", stroke="#94a3b8", strokeWidth=1.2, size=52)
            .encode(
                x=alt.X(f"{treatment_col}:N", sort=order, scale=alt.Scale(domain=order), axis=axis),
                xOffset=alt.XOffset("replicate_index:O"),
                y=alt.Y("value:Q"),
                tooltip=[
                    alt.Tooltip(f"{treatment_col}:N", title="Treatment"),
                    alt.Tooltip("value:Q", title="Value"),
                ],
            )
        )
    return alt.layer(*layers).properties(width=width, height=height, title=y_title)


def _display_column_title(value: str) -> str:
    return str(value).replace("_", " ").strip().capitalize()


def _time_tooltips(alt: Any, *, time_col: str, treatment_col: str) -> list[Any]:
    return [
        alt.Tooltip(f"{time_col}:Q", title="Time (h)"),
        alt.Tooltip(f"{treatment_col}:N", title="Treatment"),
        alt.Tooltip("y_mean:Q", title="Mean"),
        alt.Tooltip("y_sd:Q", title="SD"),
        alt.Tooltip("y_n:Q", title="N"),
    ]
