from __future__ import annotations

from reader_workbench.domains.cytometry.analysis.workflow import run_cytometry_gating
from reader_workbench.plugins.plot.cytometry_diagnostic import CytometryDiagnosticCfg, CytometryDiagnosticPlot
from reader_workbench.tests.domains.cytometry.analysis.test_gating_workflow import request, tidy_events


def test_diagnostic_plugin_wraps_domain_figure_for_shared_plot_sink() -> None:
    original = tidy_events()
    result = run_cytometry_gating(original, request())

    figures = CytometryDiagnosticPlot().render(
        None,
        {
            "original_events": original,
            "gate_definition": result.gate_definition.to_pandas(),
            "gated_events": result.gated_events.to_pandas(),
        },
        CytometryDiagnosticCfg(format=["png", "pdf"]),
    )

    assert [(figure.filename, figure.ext) for figure in figures] == [
        ("cytometry_diagnostic", "png"),
        ("cytometry_diagnostic", "pdf"),
    ]
