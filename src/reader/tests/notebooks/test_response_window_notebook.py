from __future__ import annotations

from pathlib import Path

from reader.workbench.notebooks import response_window as response_window_notebook


def test_review_notebook_uses_one_progressive_control_surface(tmp_path: Path) -> None:
    path = response_window_notebook.write_review_notebook(tmp_path)
    source = path.read_text(encoding="utf-8")

    assert 'app = marimo.App(width="medium")' in source
    assert "control_row = mo.hstack(" in source
    assert "widths=control_widths" in source
    assert "wrap=True" in source
    assert source.count("render_review_figure(") == 1
    assert source.count("mo.ui.dropdown(") >= 6
    assert "mo.ui.radio(" not in source
    assert 'label="Review view"' in source
    assert "value=next(iter(available_view_labels))" in source
    assert "value=next(iter(VIEW_LABELS))" not in source
    assert "value=next(iter(VIEW_LABELS.values()))" not in source
    assert "SFXI" not in source
    assert "verify_response_window_bundle(bundle_root)" in source
    assert "plt.close(figure)" in source
    assert 'bundle_manifest["display"]' in source
    assert "experiment_display_title_from_config" in source
    assert "experiment_selector_options(experiment_titles)" in source
    assert "expected_experiment_id=experiment_id" in source
    assert "experiment_titles[experiment.value]" in source
    assert "value=next(iter(experiment_options))" in source
    assert 'label="Reader design"' in source
    assert 'label="Condition"' in source
    assert "response_window_review_collection" in source
    assert "multi_experiment_entity_options()" in source
    assert "review_design_get, review_design_set = mo.state(None)" in source
    assert "review_summary_get, review_summary_set = mo.state(None)" in source
    assert source.count("review_design_get()") == 2
    assert source.count("on_change=review_design_set") == 2
    assert source.count("on_change=review_summary_set") == 1
    assert "retained_review_option_key" in source
    assert "retained_review_selection" in source
    assert "available_view_labels" in source
    assert "Response-window review contains no Reader design seen in multiple experiments" not in source
    assert "experiments_for_entity(active_design_id)" in source
    assert 'view_contract.selection_scope == "multi_experiment_design"' in source
    assert 'view_contract.selection_scope == "review_collection"' in source
    assert 'view_contract.reduction_mode == "selected"' in source
    assert 'view_contract.condition_mode == "selected"' in source
    assert 'bundle_manifest["source_records"]' in source
    assert 'label="Response summary"' in source
    assert "response_summary_options(available_reductions)" in source
    assert "common_cross_experiment_reductions(" in source
    assert "Reference examples across" not in source
    assert "review_view_spec" in source
    assert "Evidence interpretation" in source
    assert "Figure description" in source
    assert "Experiment coverage" in source
    assert "Bundle provenance" in source
    assert "lazy=True" in source
    assert "from reader.api.response_window.review import" in source
    assert "Connect growth and fluorescence after" in source
    assert "Response state `r_i`" not in source
    assert "Genotype" not in source
    assert "candidate" not in source.lower()


def test_review_notebook_generation_does_not_query_marimo_distribution(monkeypatch, tmp_path: Path) -> None:
    def unexpected_version_lookup(_distribution: str) -> str:
        raise AssertionError("review notebook generation must not query optional distribution metadata")

    monkeypatch.setattr(response_window_notebook, "version", unexpected_version_lookup, raising=False)

    path = response_window_notebook.write_review_notebook(tmp_path)

    assert '__generated_with = "0.23.14"' in path.read_text(encoding="utf-8")
