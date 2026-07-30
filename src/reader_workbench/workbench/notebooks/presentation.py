from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from reader_workbench.workbench.config import load_reader_config_document

_DATE_PREFIX = re.compile(r"^(?P<date>\d{8})(?:[_-]+(?P<name>.+))?$")
_WORD_BREAK = re.compile(r"[_-]+")
_UPPERCASE_TOKENS = frozenset(
    {
        "cfp",
        "dna",
        "gfp",
        "iptg",
        "od600",
        "rfp",
        "rna",
        "yfp",
    }
)


def experiment_display_title(*, experiment_id: str, authored_title: str | None = None) -> str:
    """Resolve a human title without changing the stable experiment identity."""

    stable_id = _required_text(experiment_id, field="experiment_id")
    title = str(authored_title or "").strip()
    if title and title != stable_id:
        return title

    match = _DATE_PREFIX.fullmatch(stable_id)
    if match is None:
        return _humanize_slug(stable_id)

    date_label = _date_label(match.group("date"))
    name = str(match.group("name") or "").strip()
    if not name:
        return date_label
    return f"{date_label} · {_humanize_slug(name)}"


def experiment_display_title_from_config(
    config_path: Path,
    *,
    expected_experiment_id: str | None = None,
) -> str:
    """Resolve the authored or fallback title from a Reader experiment config."""

    payload = load_reader_config_document(Path(config_path))
    experiment = payload.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment config must define an experiment mapping.")
    experiment_id = _required_text(experiment.get("id"), field="experiment.id")
    authored_title = experiment.get("title")
    if authored_title is not None and not isinstance(authored_title, str):
        raise ValueError("experiment.title must be a string when provided.")
    if expected_experiment_id is not None and experiment_id != expected_experiment_id:
        raise ValueError(
            "experiment config identity disagrees with the expected experiment ID: "
            f"{experiment_id!r} != {expected_experiment_id!r}."
        )
    return experiment_display_title(
        experiment_id=experiment_id,
        authored_title=authored_title,
    )


def experiment_selector_options(experiment_titles: Mapping[str, str]) -> dict[str, str]:
    """Map readable selector labels to stable experiment IDs."""

    normalized: dict[str, str] = {}
    for raw_experiment_id, title in experiment_titles.items():
        experiment_id = _required_text(raw_experiment_id, field="experiment_id")
        if experiment_id in normalized:
            raise ValueError(f"duplicate experiment IDs after normalization: {experiment_id!r}.")
        normalized[experiment_id] = _required_text(title, field="experiment_title")
    title_counts = Counter(normalized.values())
    options: dict[str, str] = {}
    for experiment_id in sorted(normalized):
        title = normalized[experiment_id]
        label = title if title_counts[title] == 1 else f"{title} · {_identity_suffix(experiment_id)}"
        if label in options:
            label = f"{title} · {experiment_id}"
        options[label] = experiment_id
    if len(options) != len(normalized):
        raise ValueError("experiment selector labels must be unique.")
    return options


def _humanize_slug(value: str) -> str:
    words = [word for word in _WORD_BREAK.split(value.strip()) if word]
    if not words:
        raise ValueError("experiment_id must contain displayable text.")
    return " ".join(_display_word(word) for word in words)


def _display_word(word: str) -> str:
    lowered = word.casefold()
    if lowered in _UPPERCASE_TOKENS:
        return word.upper()
    if any(character.isupper() for character in word) and any(character.islower() for character in word):
        return word
    if any(character.isdigit() for character in word):
        return word.upper()
    return word[:1].upper() + word[1:]


def _identity_suffix(experiment_id: str) -> str:
    match = _DATE_PREFIX.fullmatch(experiment_id)
    return _date_label(match.group("date")) if match is not None else experiment_id


def _date_label(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d")


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required for notebook presentation.")
    return text


__all__ = [
    "experiment_display_title",
    "experiment_display_title_from_config",
    "experiment_selector_options",
]
