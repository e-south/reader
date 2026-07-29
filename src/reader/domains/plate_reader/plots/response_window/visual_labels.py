"""Display-only labels for response-window review figures."""

from __future__ import annotations

from collections.abc import Mapping
from textwrap import wrap

from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER

STATE_COLORS = {
    "00": "#374151",
    "10": "#0f766e",
    "01": "#2563eb",
    "11": "#be123c",
}
STATE_MARKERS = {"00": "o", "10": "s", "01": "^", "11": "D"}


def state_labels(display: Mapping[str, object]) -> dict[str, str]:
    value = display.get("state_labels")
    if not isinstance(value, Mapping):
        raise ValueError("validated display state labels must be a mapping.")
    labels = {str(key): str(label) for key, label in value.items()}
    if set(labels) != set(STATE_ORDER):
        raise ValueError(f"display state labels must preserve state order {list(STATE_ORDER)}.")
    return {state: labels[state] for state in STATE_ORDER}


def channels(display: Mapping[str, object]) -> dict[str, str]:
    value = display.get("channels")
    if not isinstance(value, Mapping):
        raise ValueError("validated display channels must be a mapping.")
    return {str(key): str(label) for key, label in value.items()}


def condition_ticks(
    display: Mapping[str, object],
    *,
    include_codes: bool = True,
    width: int = 16,
) -> list[str]:
    labels = state_labels(display)
    result: list[str] = []
    for state in STATE_ORDER:
        lines = wrap(labels[state], width=width, break_long_words=False, break_on_hyphens=False)
        if include_codes:
            lines.append(f"({state})")
        result.append("\n".join(lines))
    return result


def response_axis_label(display: Mapping[str, object]) -> str:
    return f"log2[({response_ratio_label(display)})_design]"


def response_ratio_label(display: Mapping[str, object]) -> str:
    return _spaced(channels(display)["response_ratio"])


def magnitude_ratio_label(display: Mapping[str, object]) -> str:
    return _spaced(channels(display)["magnitude_ratio"])


def anchored_fluorescence_axis_label(display: Mapping[str, object]) -> str:
    values = channels(display)
    ratio = magnitude_ratio_label(display)
    reference = values["reference_design_id"]
    return f"log2[({ratio})_design / ({ratio})_{reference}]"


def response_uncertainty_axis_label(display: Mapping[str, object]) -> str:
    return f"90th percentile uncertainty\n({response_axis_label(display)})"


def anchored_fluorescence_uncertainty_axis_label(display: Mapping[str, object]) -> str:
    reference = channels(display)["reference_design_id"]
    return f"90th percentile uncertainty\n({reference}-relative log2 fluorescence)"


def component_ticks(display: Mapping[str, object]) -> list[str]:
    labels = state_labels(display)
    response_ratio = response_ratio_label(display)
    magnitude_ratio = magnitude_ratio_label(display)
    reference = channels(display)["reference_design_id"]
    return [
        *(f"{response_ratio} (design)\n{labels[state]}" for state in STATE_ORDER),
        *(f"{reference}-relative {magnitude_ratio}\n{labels[state]}" for state in STATE_ORDER),
    ]


def reduction_label(row: Mapping[str, object]) -> str:
    start = float(row["window_start_event_h"])
    end = float(row["window_end_event_h"])
    method_label, basis_label, role_label = _reduction_terms(row)
    return f"{start:g}-{end:g} h {method_label}\n{basis_label}; {role_label}"


def response_summary_label(row: Mapping[str, object]) -> str:
    """Return a compact human label while stable IDs retain full semantics."""

    start = float(row["window_start_event_h"])
    end = float(row["window_end_event_h"])
    method_label, basis_label, role_label = _reduction_terms(row)
    basis_prefix = "" if basis_label == "post-event" else "pre-adjusted "
    return f"{start:g}-{end:g} h {basis_prefix}{method_label} ({role_label})"


def _reduction_terms(row: Mapping[str, object]) -> tuple[str, str, str]:
    method = str(row["reduction_method"])
    basis = str(row["response_basis"])
    role = str(row["reduction_role"])
    method_label = {"geometric_time_mean": "log mean", "integrated_linear_mean": "linear AUC"}.get(method)
    basis_label = {"post_window": "post-event", "post_minus_pre": "post minus pre-event"}.get(basis)
    role_label = {"primary": "primary", "sensitivity": "sensitivity"}.get(role)
    if method_label is None or basis_label is None or role_label is None:
        raise ValueError(
            f"unsupported response-window reduction terms: method={method!r}, basis={basis!r}, role={role!r}."
        )
    return method_label, basis_label, role_label


def _spaced(value: object) -> str:
    return str(value).replace("/", " / ")


__all__ = [
    "STATE_COLORS",
    "STATE_MARKERS",
    "anchored_fluorescence_axis_label",
    "channels",
    "component_ticks",
    "condition_ticks",
    "magnitude_ratio_label",
    "reduction_label",
    "response_axis_label",
    "response_ratio_label",
    "response_summary_label",
    "state_labels",
]
