"""Generate the response-window Marimo handoff review surface."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path


def write_review_notebook(out_dir: Path) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "review.py"
    source = _NOTEBOOK_SOURCE.replace("__MARIMO_VERSION__", version("marimo"))
    path.write_text(source, encoding="utf-8")
    if path.stat().st_size == 0:
        raise RuntimeError("generated response-window review notebook is empty.")
    return path


_NOTEBOOK_SOURCE = '''import marimo

__generated_with = "__MARIMO_VERSION__"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import marimo as mo

    from reader.response_window_review import (
        VIEW_LABELS,
        load_review_tables,
        measured_response_example_rows,
        render_review_figure,
        response_summary_options,
        review_view_spec,
        selected_handoff_row,
    )
    from reader.response_window import verify_response_window_bundle

    return Path, VIEW_LABELS, load_review_tables, measured_response_example_rows, mo, plt, render_review_figure, response_summary_options, review_view_spec, selected_handoff_row, verify_response_window_bundle


@app.cell
def _(Path, load_review_tables, verify_response_window_bundle):
    bundle_root = Path(__file__).resolve().parent
    bundle = verify_response_window_bundle(bundle_root)
    bundle_manifest = bundle.manifest
    design_rows, well_rows, trace_rows, event_rows = load_review_tables(bundle_root)
    return bundle_manifest, bundle_root, design_rows, event_rows, trace_rows, well_rows


@app.cell
def _(bundle_manifest):
    display_contract = bundle_manifest["display"]
    return (display_contract,)


@app.cell
def _(design_rows, mo):
    experiment_options = sorted(design_rows["experiment_id"].astype(str).unique().tolist())
    if not experiment_options:
        raise ValueError("Response-window bundle contains no experiments.")
    experiment = mo.ui.dropdown(
        options=experiment_options,
        value=experiment_options[0],
        label="Experiment",
        searchable=True,
        full_width=True,
    )
    return experiment, experiment_options


@app.cell
def _(design_rows, experiment, mo):
    design_options = sorted(
        design_rows.loc[
            design_rows["experiment_id"].astype(str).eq(experiment.value),
            "design_id",
        ].astype(str).unique().tolist()
    )
    if not design_options:
        raise ValueError(f"Experiment {experiment.value!r} contains no designs.")
    design = mo.ui.dropdown(
        options=design_options,
        value=design_options[0],
        label="Design",
        searchable=True,
        full_width=True,
    )
    return design, design_options


@app.cell
def _(design, design_rows, experiment, mo, response_summary_options):
    available_reductions = design_rows.loc[
        design_rows["experiment_id"].astype(str).eq(experiment.value)
        & design_rows["design_id"].astype(str).eq(design.value)
    ].copy()
    reduction_options = response_summary_options(available_reductions)
    if not reduction_options:
        raise ValueError("Selected design contains no response-window reductions.")
    reduction = mo.ui.dropdown(
        options=reduction_options,
        value=next(iter(reduction_options)),
        label="Response summary",
        full_width=True,
    )
    return available_reductions, reduction, reduction_options


@app.cell
def _(VIEW_LABELS, mo):
    view = mo.ui.dropdown(
        options=VIEW_LABELS,
        value=next(iter(VIEW_LABELS)),
        label="Review view",
        full_width=True,
    )
    return (view,)


@app.cell
def _(design, display_contract, experiment, mo, reduction, view):
    _channels = display_contract["channels"]
    introduction = mo.md(
        f"""
        # {display_contract['study_label']}

        Review how promoter growth and fluorescence trajectories after {str(display_contract['event_label']).lower()}
        are reduced into four condition-specific `{_channels['response_ratio']}` responses and four
        `{_channels['reference_design_id']}`-relative `{_channels['magnitude_ratio']}` fluorescence values.
        """
    )
    control_items = [view, experiment, design, reduction]
    control_widths = [1.1, 1.4, 1.0, 1.35]
    if view.value == "measured_response_examples":
        control_items = [view, reduction]
        control_widths = [1.1, 1.35]
    elif view.value == "reduction_sensitivity":
        control_items = [view, experiment, design]
        control_widths = [1.1, 1.4, 1.0]
    control_row = mo.hstack(
        control_items,
        widths=control_widths,
        gap=0.75,
        align="end",
        wrap=True,
    )
    controls = mo.vstack([introduction, control_row], gap=0.75)
    controls
    return (controls,)


@app.cell
def _(
    design,
    design_rows,
    display_contract,
    event_rows,
    experiment,
    mo,
    plt,
    reduction,
    measured_response_example_rows,
    render_review_figure,
    review_view_spec,
    selected_handoff_row,
    trace_rows,
    view,
    well_rows,
    bundle_manifest,
):
    active_reduction = reduction.value
    if view.value == "measured_response_examples":
        selected = measured_response_example_rows(
            design_rows,
            display=display_contract,
            reduction_id=active_reduction,
        )
    else:
        selected = selected_handoff_row(
            design_rows,
            experiment_id=experiment.value,
            design_id=design.value,
            reduction_id=active_reduction,
        )
    figure = render_review_figure(
        view_id=view.value,
        experiment_id=experiment.value,
        design_id=design.value,
        reduction_id=active_reduction,
        designs=design_rows,
        wells=well_rows,
        traces=trace_rows,
        events=event_rows,
        display=display_contract,
    )
    view_contract = review_view_spec(view.value, display=display_contract)
    evidence_columns = [
        "experiment_id",
        "design_id",
        "reference_design_id",
        "reduction_id",
        "reduction_method",
        "response_basis",
        "reduction_role",
        "event_time_estimate_assay_h",
        "event_time_uncertainty_h",
        "window_start_event_h",
        "window_end_event_h",
        "min_replicates_per_state",
        "min_observed_points_per_trace",
        "max_interior_gap_h",
        "min_pre_observed_points_per_trace",
        "max_pre_interior_gap_h",
        "r00",
        "r10",
        "r01",
        "r11",
        "b00",
        "b10",
        "b01",
        "b11",
    ]
    if view.value == "measured_response_examples":
        evidence_columns = ["example_label", "example_role", *evidence_columns]
    evidence = mo.accordion(
        {
            "Handoff row": mo.ui.dataframe(selected.loc[:, evidence_columns]),
            "Evidence interpretation": mo.md(
                f"""
                **Premise:** {view_contract.premise}

                **Decision value:** {view_contract.decision_value}

                **How to read it:** {view_contract.interpretation}

                **Figure description:** {view_contract.alt_text}

                **Limit:** {view_contract.non_claim_boundary}

                """
            ),
        }
    )
    figure_view = mo.as_html(figure).style({"max-width": "100%", "overflow": "hidden"})
    figure_description = mo.md(f"*Figure description:* {view_contract.alt_text}")
    plt.close(figure)
    mo.vstack([figure_view, figure_description, evidence], gap=1.0)
    return evidence, evidence_columns, figure_description, figure_view, selected


if __name__ == "__main__":
    app.run()
'''


__all__ = ["write_review_notebook"]
