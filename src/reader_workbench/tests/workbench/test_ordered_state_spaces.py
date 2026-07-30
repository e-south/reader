from __future__ import annotations

from pathlib import Path

import pytest

from reader_workbench.errors import ConfigError
from reader_workbench.tests.support import base_reader_config, load_models, write_config


def _config(*, state_order: list[str], values: dict[str, str], case_sensitive: bool = True) -> dict:
    return base_reader_config(
        experiment_id="exp_states",
        annotations={
            "ordered_state_spaces": {
                "four_state_conditions": {
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
                "00": "Baseline",
                "10": "Condition A",
                "01": "Condition B",
                "11": "Conditions A and B",
            },
            case_sensitive=False,
        ),
    )

    spec, declaration = load_models(path)
    wire = spec.annotations.ordered_state_spaces["four_state_conditions"]
    resolved = declaration.experiment_semantics.annotations.resolve_ordered_state_space(ref="four_state_conditions")

    assert wire.state_order == ["00", "10", "01", "11"]
    assert resolved.state_ids == ("00", "10", "01", "11")
    assert resolved.source_values == {
        "00": "Baseline",
        "10": "Condition A",
        "01": "Condition B",
        "11": "Conditions A and B",
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


@pytest.mark.parametrize("space_id", ["", " four_state_conditions ", 7])
def test_ordered_state_space_rejects_nonexact_space_ids(tmp_path: Path, space_id: object) -> None:
    data = _config(state_order=["off"], values={"off": "none"})
    space = data["annotations"]["ordered_state_spaces"].pop("four_state_conditions")
    data["annotations"]["ordered_state_spaces"] = {space_id: space}
    path = write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="keys must be non-empty, already-trimmed strings"):
        load_models(path)


def test_ordered_state_space_rejects_space_ids_that_collide_when_stringified(tmp_path: Path) -> None:
    data = _config(state_order=["off"], values={"off": "none"})
    space = data["annotations"]["ordered_state_spaces"].pop("four_state_conditions")
    data["annotations"]["ordered_state_spaces"] = {
        7: space,
        "7": space,
    }
    path = write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="keys must be non-empty, already-trimmed strings"):
        load_models(path)


def test_ordered_state_space_rejects_nonstring_value_keys(tmp_path: Path) -> None:
    data = _config(state_order=["0"], values={"0": "none"})
    data["annotations"]["ordered_state_spaces"]["four_state_conditions"]["values"] = {0: "none"}
    path = write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="values keys must be strings"):
        load_models(path)


def test_ordered_state_space_rejects_value_keys_that_collide_when_stringified(tmp_path: Path) -> None:
    data = _config(state_order=["0"], values={"0": "none"})
    data["annotations"]["ordered_state_spaces"]["four_state_conditions"]["values"] = {
        0: "first",
        "0": "second",
    }
    path = write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="values keys must be strings"):
        load_models(path)


def test_ordered_state_space_rejects_state_ids_that_require_trimming(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        _config(
            state_order=[" off "],
            values={"off": "none"},
        ),
    )

    with pytest.raises(ConfigError, match="state_order must contain non-empty, already-trimmed strings"):
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
    resolved = declaration.experiment_semantics.annotations.resolve_ordered_state_space(ref="four_state_conditions")

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
