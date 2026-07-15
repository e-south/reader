from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reader.notebook_presentation import (
    experiment_display_title,
    experiment_display_title_from_config,
    experiment_selector_options,
)


def test_experiment_display_title_prefers_authored_metadata() -> None:
    assert (
        experiment_display_title(
            experiment_id="20260706_sfxi_sensor-panel-m9-glu-secg",
            authored_title="SECG promoter sensor panel in M9 glucose",
        )
        == "SECG promoter sensor panel in M9 glucose"
    )


def test_experiment_display_title_humanizes_stable_id_as_fallback() -> None:
    assert (
        experiment_display_title(
            experiment_id="20260706_sfxi_sensor-panel-m9-glu-secg",
            authored_title=None,
        )
        == "2026-07-06 · SFXI Sensor Panel M9 Glu SECG"
    )


def test_experiment_display_title_preserves_mixed_case_biological_identifiers() -> None:
    assert (
        experiment_display_title(
            experiment_id="20260117_sfxi_ref-pDual10-pES1-Eco1",
            authored_title=None,
        )
        == "2026-01-17 · SFXI Ref pDual10 pES1 Eco1"
    )


def test_experiment_display_title_from_config_checks_identity(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema": "reader/v7",
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
            "20260117_sfxi_ref-pDual10": "Reference promoter",
            "20260121_sfxi_ref-pDual10": "Reference promoter",
        }
    ) == {
        "Reference promoter · 2026-01-17": "20260117_sfxi_ref-pDual10",
        "Reference promoter · 2026-01-21": "20260121_sfxi_ref-pDual10",
    }
