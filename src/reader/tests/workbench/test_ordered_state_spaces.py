from __future__ import annotations

from pathlib import Path

import pytest

from reader.errors import ConfigError
from reader.tests.support import base_reader_config, load_models, write_config


def _config(*, state_order: list[str], values: dict[str, str], case_sensitive: bool = True) -> dict:
    return base_reader_config(
        experiment_id="exp_states",
        annotations={
            "ordered_state_spaces": {
                "stress_states": {
                    "column": "treatment",
                    "state_order": state_order,
                    "values": values,
                    "case_sensitive": case_sensitive,
                }
            }
        },
    )


def test_ordered_state_space_resolves_declared_order_and_exact_values(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        _config(
            state_order=["00", "10", "01", "11"],
            values={
                "00": "No stress",
                "10": "Ethanol",
                "01": "Ciprofloxacin",
                "11": "Ethanol + Ciprofloxacin",
            },
            case_sensitive=False,
        ),
    )

    spec, declaration = load_models(path)
    wire = spec.annotations.ordered_state_spaces["stress_states"]
    resolved = declaration.experiment_semantics.annotations.resolve_ordered_state_space(ref="stress_states")

    assert wire.state_order == ["00", "10", "01", "11"]
    assert resolved.state_ids == ("00", "10", "01", "11")
    assert resolved.source_values == {
        "00": "No stress",
        "10": "Ethanol",
        "01": "Ciprofloxacin",
        "11": "Ethanol + Ciprofloxacin",
    }
    assert resolved.column == "treatment"
    assert resolved.case_sensitive is False


def test_ordered_state_space_rejects_duplicate_state_ids(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        _config(
            state_order=["00", "00"],
            values={"00": "none"},
        ),
    )

    with pytest.raises(ConfigError, match="state ids must be unique"):
        load_models(path)


def test_ordered_state_space_rejects_case_insensitive_source_collisions(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        _config(
            state_order=["off", "on"],
            values={"off": "Untreated", "on": " untreated "},
            case_sensitive=False,
        ),
    )

    with pytest.raises(ConfigError, match="source values must be unique under case_sensitive=false"):
        load_models(path)


def test_ordered_state_space_allows_case_distinct_source_values_when_case_sensitive(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        _config(
            state_order=["lower", "title"],
            values={"lower": "untreated", "title": "Untreated"},
            case_sensitive=True,
        ),
    )

    _, declaration = load_models(path)
    resolved = declaration.experiment_semantics.annotations.resolve_ordered_state_space(ref="stress_states")

    assert resolved.source_values == {"lower": "untreated", "title": "Untreated"}


def test_ordered_state_space_rejects_empty_state_list(tmp_path: Path) -> None:
    path = write_config(tmp_path, _config(state_order=[], values={}))

    with pytest.raises(ConfigError, match="state_order must be a non-empty list"):
        load_models(path)


def test_ordered_state_space_rejects_unknown_reference(tmp_path: Path) -> None:
    path = write_config(tmp_path, _config(state_order=["off"], values={"off": "none"}))
    _, declaration = load_models(path)

    with pytest.raises(ValueError, match="Unknown state_map_ref 'missing'.*annotations.ordered_state_spaces"):
        declaration.experiment_semantics.annotations.resolve_ordered_state_space(ref="missing")
