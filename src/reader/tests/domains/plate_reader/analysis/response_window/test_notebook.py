from __future__ import annotations

from pathlib import Path

from reader.domains.plate_reader.analysis.response_window import notebook as response_window_notebook


def test_review_notebook_uses_one_progressive_control_surface(tmp_path: Path) -> None:
    path = response_window_notebook.write_review_notebook(tmp_path)
    source = path.read_text(encoding="utf-8")

    assert 'app = marimo.App(width="medium")' in source
    assert "control_row = mo.hstack(" in source
    assert "widths=control_widths" in source
    assert "wrap=True" in source
    assert "control_widths = [1.1, 1.4, 1.0, 1.35]" in source
    assert source.count("render_review_figure(") == 1
    assert source.count("mo.ui.dropdown(") == 4
    assert "mo.ui.radio(" not in source
    assert 'label="Review view"' in source
    assert "value=next(iter(VIEW_LABELS))" in source
    assert "value=next(iter(VIEW_LABELS.values()))" not in source
    assert "SFXI" not in source
    assert "verify_response_window_bundle(bundle_root)" in source
    assert "plt.close(figure)" in source
    assert 'bundle_manifest["display"]' in source
    assert 'view.value == "measured_response_examples"' in source
    assert 'label="Response summary"' in source
    assert "response_summary_options(available_reductions)" in source
    assert "Reference examples across" not in source
    assert "review_view_spec" in source
    assert "Evidence interpretation" in source
    assert "Figure description" in source
    assert "from reader.response_window_review import" in source
    assert "Review how promoter growth and fluorescence trajectories after" in source
    assert "Response state `r_i`" not in source
    assert "Reader does not apply a campaign target mask or calculate an OPAL score" not in source


def test_review_notebook_generation_does_not_query_marimo_distribution(monkeypatch, tmp_path: Path) -> None:
    def unexpected_version_lookup(_distribution: str) -> str:
        raise AssertionError("review notebook generation must not query optional distribution metadata")

    monkeypatch.setattr(response_window_notebook, "version", unexpected_version_lookup, raising=False)

    path = response_window_notebook.write_review_notebook(tmp_path)

    assert '__generated_with = "0.23.14"' in path.read_text(encoding="utf-8")
