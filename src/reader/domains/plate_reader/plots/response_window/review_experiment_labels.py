"""Compact Reader experiment labels for response-window figures."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime

_DATE_PREFIX = re.compile(r"^(?P<date>\d{8})(?:[_-]|$)")


def compact_experiment_plot_labels(
    experiment_order: Sequence[str],
    labels: Mapping[str, str],
) -> dict[str, str]:
    """Prefer compact dates while preserving unique deterministic labels."""

    candidates: dict[str, str] = {}
    for experiment_id in experiment_order:
        match = _DATE_PREFIX.match(experiment_id)
        if match is None:
            candidates[experiment_id] = labels[experiment_id]
            continue
        try:
            candidates[experiment_id] = datetime.strptime(match.group("date"), "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            candidates[experiment_id] = labels[experiment_id]
    counts = Counter(candidates.values())
    return {
        experiment_id: candidate if counts[candidate] == 1 else f"{candidate} · E{index + 1}"
        for index, (experiment_id, candidate) in enumerate(candidates.items())
    }


__all__ = ["compact_experiment_plot_labels"]
