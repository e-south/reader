from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from reader.domains.plate_reader.ordering import order_levels
from reader.domains.plate_reader.plots.grouping import GroupMatch, resolve_groups

_PanelBy = Literal["channel", "x", "group"]


def build_figure_groups(
    *,
    stats: pd.DataFrame,
    group_col: str | None,
    panel_by: _PanelBy,
    pool_sets: list[dict[str, list[str]]] | None,
    pool_match: GroupMatch,
) -> list[tuple[str, list[str | None]]]:
    if not group_col:
        return [("all", [None])]

    universe = order_levels(stats[group_col].astype(str).unique().tolist())
    if panel_by == "group":
        if pool_sets:
            resolved = resolve_groups(universe, pool_sets, match=pool_match)
            ordered: list[str] = []
            seen: set[str] = set()
            for _, values in resolved:
                for value in values:
                    if value in universe and value not in seen:
                        ordered.append(value)
                        seen.add(value)
            return [("all", ordered or universe)]
        return [("all", universe)]

    if pool_sets:
        resolved = resolve_groups(universe, pool_sets, match=pool_match)
        union: list[str] = []
        seen: set[str] = set()
        for _, values in resolved:
            for value in values:
                if value in universe and value not in seen:
                    union.append(value)
                    seen.add(value)
        return [(group, [group]) for group in (union or universe)]
    return [(group, [group]) for group in universe]


def resolve_panel_configuration(
    *,
    panel_by: _PanelBy,
    members: list[str | None],
    y_list: list[str],
    group_col: str | None,
    channel_select: str | None,
    selected_channel_default: str | None,
    ch_for_file: str | None,
    stats: pd.DataFrame,
    x_col: str,
) -> tuple[list[str], str | None]:
    if panel_by == "channel":
        return y_list, None
    if panel_by == "group":
        if not group_col:
            raise ValueError("panel_by='group' requires 'group_on'")
        selected_channel = (
            (str(ch_for_file) if ch_for_file is not None else None) or channel_select or selected_channel_default
        )
        if not selected_channel:
            raise ValueError(
                "panel_by='group' requires an explicit channel: set 'channel_select' or pass a single string to 'y'."
            )
        return [str(member) for member in members], str(selected_channel)

    selected_channel = channel_select or selected_channel_default
    if not selected_channel:
        raise ValueError(
            "panel_by='x' requires an explicit channel: set 'channel_select' or pass a single string to 'y'."
        )
    sub_stats = stats[stats["channel"].astype(str) == str(selected_channel)]
    x_levels = sub_stats[x_col].astype(str).unique().tolist()
    return order_levels(x_levels), str(selected_channel)


def compute_shared_ylim(
    *,
    stats: pd.DataFrame,
    snapped: pd.DataFrame,
    panels: list[str],
    panel_by: _PanelBy,
    selected_channel: str,
    group_col: str | None,
    x_col: str,
    agg: str,
    err: str,
) -> tuple[float | None, float | None]:
    if panel_by not in {"group", "x"} or len(panels) <= 1:
        return None, None

    def _collect_limits_for_subset(
        sbar_sub: pd.DataFrame,
        srep_sub: pd.DataFrame,
    ) -> tuple[float | None, float | None]:
        values = (
            pd.to_numeric(sbar_sub[agg], errors="coerce").dropna() if not sbar_sub.empty else pd.Series([], dtype=float)
        )
        value_min = float(values.min()) if not values.empty else None
        value_max = float(values.max()) if not values.empty else None
        if err == "sem" and "sem" in sbar_sub.columns:
            top = (
                pd.to_numeric(sbar_sub[agg], errors="coerce") + pd.to_numeric(sbar_sub["sem"], errors="coerce")
            ).dropna()
            if not top.empty:
                value_max = max(value_max or -np.inf, float(top.max()))
        elif err == "iqr" and {"q1", "q3"}.issubset(sbar_sub.columns):
            if agg == "median":
                top = pd.to_numeric(sbar_sub["q3"], errors="coerce").dropna()
                if not top.empty:
                    value_max = max(value_max or -np.inf, float(top.max()))
            else:
                half = 0.5 * (
                    pd.to_numeric(sbar_sub["q3"], errors="coerce") - pd.to_numeric(sbar_sub["q1"], errors="coerce")
                )
                top = (pd.to_numeric(sbar_sub[agg], errors="coerce") + half).dropna()
                if not top.empty:
                    value_max = max(value_max or -np.inf, float(top.max()))
        if not srep_sub.empty and "value" in srep_sub.columns:
            replicate_values = pd.to_numeric(srep_sub["value"], errors="coerce").dropna()
            if not replicate_values.empty:
                value_min = min(value_min if value_min is not None else float("inf"), float(replicate_values.min()))
                value_max = max(value_max if value_max is not None else float("-inf"), float(replicate_values.max()))
        return value_min, value_max

    low_values: list[float] = []
    high_values: list[float] = []
    for panel in panels:
        if panel_by == "group":
            sbar_sub = stats[
                (stats["channel"].astype(str) == str(selected_channel)) & (stats[group_col].astype(str) == str(panel))
            ]
            srep_sub = snapped[
                (snapped["channel"].astype(str) == str(selected_channel))
                & (snapped[group_col].astype(str) == str(panel))
            ]
        else:
            sbar_sub = stats[
                (stats["channel"].astype(str) == str(selected_channel)) & (stats[x_col].astype(str) == str(panel))
            ]
            srep_sub = snapped[
                (snapped["channel"].astype(str) == str(selected_channel)) & (snapped[x_col].astype(str) == str(panel))
            ]
        value_min, value_max = _collect_limits_for_subset(sbar_sub, srep_sub)
        if value_min is not None:
            low_values.append(value_min)
        if value_max is not None:
            high_values.append(value_max)
    if not high_values:
        return None, None
    y_lo = min(0.0, float(np.nanmin(low_values))) if low_values else 0.0
    y_hi = float(np.nanmax(high_values))
    pad = 0.05 * (y_hi - y_lo if y_hi > y_lo else max(1.0, y_hi or 1.0))
    return y_lo, y_hi + pad
