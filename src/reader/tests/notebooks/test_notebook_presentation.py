from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reader.errors import ConfigError
from reader.workbench.notebooks.presentation import (
    experiment_display_title,
    experiment_display_title_from_config,
    experiment_selector_options,
)


def test_experiment_display_title_prefers_authored_metadata() -> None:
    assert (
        experiment_display_title(
            experiment_id="20260102_sfxi_four-state-panel",
            authored_title="Two-input response panel",
        )
        == "Two-input response panel"
    )


def test_experiment_display_title_humanizes_stable_id_as_fallback() -> None:
    assert (
        experiment_display_title(
            experiment_id="20260102_sfxi_four-state-panel",
            authored_title=None,
        )
        == "2026-01-02 · Sfxi Four State Panel"
    )


def test_experiment_display_title_preserves_mixed_case_biological_identifiers() -> None:
    assert (
        experiment_display_title(
            experiment_id="20260117_sfxi_ref-ControlA-DesignB",
            authored_title=None,
        )
        == "2026-01-17 · Sfxi Ref ControlA DesignB"
    )


def test_experiment_display_title_from_config_checks_identity(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "reader/v8",
                "experiment": {"id": "exp_alpha", "title": "Alpha screen"},
                "protocol": {"id": "workbench/generic"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert experiment_display_title_from_config(config_path, expected_experiment_id="exp_alpha") == "Alpha screen"
    with pytest.raises(ValueError, match="config identity disagrees"):
        experiment_display_title_from_config(config_path, expected_experiment_id="exp_beta")


def test_experiment_display_title_from_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """schema: reader/v8
experiment:
  id: exp_alpha
  id: exp_beta
protocol:
  id: workbench/generic
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate key"):
        experiment_display_title_from_config(config_path)


def test_experiment_display_title_from_config_accepts_verified_source_snapshot(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """schema: reader/v8
experiment:
  id: exp_alpha
  title: Alpha evidence
""",
        encoding="utf-8",
    )

    assert experiment_display_title_from_config(config_path) == "Alpha evidence"


def test_experiment_selector_options_preserve_stable_ids_and_disambiguate_titles() -> None:
    assert experiment_selector_options(
        {
            "exp_alpha": "Shared title",
            "exp_beta": "Shared title",
            "exp_gamma": "Unique title",
        }
    ) == {
        "Shared title · exp_alpha": "exp_alpha",
        "Shared title · exp_beta": "exp_beta",
        "Unique title": "exp_gamma",
    }


def test_experiment_selector_options_use_stable_dates_for_repeated_titles() -> None:
    assert experiment_selector_options(
        {
            "20260117_sfxi_reference": "Reference design",
            "20260121_sfxi_reference": "Reference design",
        }
    ) == {
        "Reference design · 2026-01-17": "20260117_sfxi_reference",
        "Reference design · 2026-01-21": "20260121_sfxi_reference",
    }


def test_experiment_selector_options_reject_normalized_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate experiment IDs after normalization"):
        experiment_selector_options({"exp_alpha": "First", " exp_alpha ": "Second"})


def test_notebook_presentation_does_not_embed_study_or_metric_vocabulary() -> None:
    source = (Path(__file__).resolve().parents[2] / "workbench" / "notebooks" / "presentation.py").read_text(
        encoding="utf-8"
    )

    for forbidden in ("sfxi", "rmf", "se" + "cg"):
        assert f'"{forbidden}"' not in source.lower()
