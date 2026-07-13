from __future__ import annotations

from reader.workbench.templates import resolve_notebook_template_descriptor


def test_cytometry_template_keeps_event_analysis_in_polars_until_plot_payloads() -> None:
    template = resolve_notebook_template_descriptor("notebook/cytometry").load_body()

    assert "cytometry_analysis.scan_event_table" in template
    assert "cytometry_analysis.prepare_event_table" in template
    assert template.count("cytometry_analysis.distinct_string_values_by_column") == 1
    assert "cytometry_analysis.distinct_string_values(df" not in template
    assert "cytometry_analysis.analyze_events" in template
    assert "cytometry_analysis.prepare_plot_events" in template
    assert "cyto_event_wide = _wide.to_pandas" not in template
    assert "cyto_gated_events = _df.loc" not in template
    assert template.count(".to_pandas(use_pyarrow_extension_array=False)") == 2


def test_cytometry_template_exports_polars_statistics_through_public_api() -> None:
    template = resolve_notebook_template_descriptor("notebook/cytometry").load_body()

    assert "cyto_stats_sample.write_csv(_stats_target)" in template
    assert "cyto_stats_sample.to_csv(_stats_target, index=False)" not in template
