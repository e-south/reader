from __future__ import annotations

from dataclasses import replace

from reader_workbench.domains.cytometry.analysis.workflow import run_cytometry_gating
from reader_workbench.domains.cytometry.plots.diagnostic import render_cytometry_diagnostic
from reader_workbench.tests.domains.cytometry.analysis.test_gating_workflow import request, tidy_events


def test_diagnostic_renders_cells_singlets_fluorescence_and_retention_panels() -> None:
    original = tidy_events()
    result = run_cytometry_gating(original, request())

    figure = render_cytometry_diagnostic(
        original,
        result.gate_definition.to_pandas(),
        result.gated_events.to_pandas(),
        max_events=1_000,
        title="Configured cytometry gating",
    )

    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes] == [
        "Cells gate",
        "Singlets gate",
        "Fluorescence",
        "Final retention",
    ]


def test_diagnostic_renders_reporter_only_events_when_both_structural_gates_are_disabled() -> None:
    original = tidy_events().loc[lambda frame: frame["channel"] == "reporter"].copy()
    configured = request()
    configured = replace(
        configured,
        gate=replace(
            configured.gate,
            cells_enabled=False,
            cells_x_channel="<cells-x-channel>",
            cells_y_channel="<cells-y-channel>",
            singlets_enabled=False,
            singlet_x_channel="<singlet-denominator-channel>",
            singlet_y_channel="<singlet-numerator-channel>",
        ),
    )
    result = run_cytometry_gating(original, configured)

    figure = render_cytometry_diagnostic(
        original,
        result.gate_definition.to_pandas(),
        result.gated_events.to_pandas(),
        max_events=1_000,
    )

    assert [text.get_text() for text in figure.axes[0].texts] == ["Disabled"]
    assert [text.get_text() for text in figure.axes[1].texts] == ["Disabled"]
    assert figure.axes[2].get_xlabel() == "reporter"
