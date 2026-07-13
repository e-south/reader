from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pandas as pd
import pytest


def test_retron_aggregate_dataframe_preparation_is_domain_owned() -> None:
    domain_module = importlib.import_module("reader.domains.plate_reader.analysis.retron_review_aggregate")
    plots_module = importlib.import_module("reader.workbench.notebooks._retron_review_aggregate_plots")
    public_module = importlib.import_module("reader.workbench.notebooks.retron_review")

    old_module_path = Path(plots_module.__file__).with_name("_retron_review_aggregate_data.py")
    plots_source = inspect.getsource(plots_module)
    domain_source = inspect.getsource(domain_module)

    assert not old_module_path.exists()
    assert "reader.domains.plate_reader.analysis import retron_review_aggregate" in plots_source
    assert "_retron_review_aggregate_data" not in plots_source
    assert "reader.workbench" not in domain_source
    assert not hasattr(plots_module, "build_specificity_matrix")
    assert public_module.build_specificity_matrix is domain_module.build_specificity_matrix


def test_retron_review_semantic_helpers_are_domain_owned() -> None:
    semantics_module = importlib.import_module("reader.domains.plate_reader.analysis.retron_review_semantics")
    aggregate_module = importlib.import_module("reader.domains.plate_reader.analysis.retron_review_aggregate")
    bundle_module = importlib.import_module("reader.workbench.notebooks._retron_review_bundle")
    experiment_plots_module = importlib.import_module("reader.workbench.notebooks._retron_review_experiment_plots")
    notebook_ui_module = importlib.import_module("reader.workbench.notebooks._retron_review_notebook_ui")

    old_module_path = Path(bundle_module.__file__).with_name("_retron_review_shared.py")
    consumer_sources = (
        inspect.getsource(aggregate_module),
        inspect.getsource(bundle_module),
        inspect.getsource(experiment_plots_module),
        inspect.getsource(notebook_ui_module),
    )

    assert not old_module_path.exists()
    assert all("_retron_review_shared" not in source for source in consumer_sources)
    assert "reader.domains.plate_reader.analysis import retron_review_semantics" in consumer_sources[0]
    assert "def _sponge_sort_key" not in consumer_sources[0]
    assert semantics_module.sponge_sort_key("CpxR-LexA") == (1, 2, "CpxR-LexA")


@pytest.mark.parametrize("relevant_sensor_pair", [True, False])
def test_expected_vs_observed_returns_typed_empty_frame_for_mono_only_summary(
    relevant_sensor_pair: bool,
) -> None:
    aggregate_module = importlib.import_module("reader.domains.plate_reader.analysis.retron_review_aggregate")
    summary = pd.DataFrame(
        {
            "sensor": ["CpxR"],
            "sponge": ["CpxR"],
            "metric": ["S_AUC"],
            "value": [1.25],
            "relevant_sensor_pair": [relevant_sensor_pair],
            "is_relevant_stress": [True],
            "sponge_family_size": ["mono"],
        }
    )

    result = aggregate_module.build_expected_vs_observed_frame(
        summary,
        sensor_target_map={"CpxR": ("CpxR",)},
    )

    assert result.empty
    assert result.columns.tolist() == [
        "sensor",
        "sponge",
        "observed",
        "expected_best_single",
        "expected_sum",
        "relevant_motif_count",
        "sponge_family_size",
    ]
