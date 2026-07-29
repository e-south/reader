"""Generate the response-window Marimo handoff review surface."""

from __future__ import annotations

from pathlib import Path

_MARIMO_NOTEBOOK_FORMAT_VERSION = "0.23.14"


def write_review_notebook(out_dir: Path) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "review.py"
    source = _NOTEBOOK_SOURCE.replace("__MARIMO_VERSION__", _MARIMO_NOTEBOOK_FORMAT_VERSION)
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

    from reader.domains.review import retained_review_option_key, retained_review_selection
    from reader.workbench.notebooks.presentation import (
        experiment_display_title_from_config,
        experiment_selector_options,
    )
    from reader.response_window import verify_response_window_bundle
    from reader.response_window_review import (
        VIEW_LABELS,
        common_cross_experiment_reductions,
        cross_experiment_design_rows,
        load_review_tables,
        measured_response_example_rows,
        render_review_figure,
        response_summary_options,
        response_window_review_collection,
        review_view_spec,
        selected_handoff_row,
    )

    return Path, VIEW_LABELS, common_cross_experiment_reductions, cross_experiment_design_rows, experiment_display_title_from_config, experiment_selector_options, load_review_tables, measured_response_example_rows, mo, plt, render_review_figure, response_summary_options, response_window_review_collection, retained_review_option_key, retained_review_selection, review_view_spec, selected_handoff_row, verify_response_window_bundle


@app.cell
def _(mo):
    review_design_get, review_design_set = mo.state(None)
    return review_design_get, review_design_set


@app.cell
def _(mo):
    review_summary_get, review_summary_set = mo.state(None)
    return review_summary_get, review_summary_set


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
def _(
    bundle_manifest,
    bundle_root,
    design_rows,
    display_contract,
    experiment_display_title_from_config,
    experiment_selector_options,
):
    _source_records = bundle_manifest["source_records"]
    if not isinstance(_source_records, list) or not _source_records:
        raise ValueError("Verified response-window bundle contains no source records.")
    experiment_ids = tuple(str(record["experiment_id"]) for record in _source_records)
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("Verified response-window bundle contains duplicate experiment IDs.")
    _observed_experiments = set(design_rows["experiment_id"].astype(str))
    if _observed_experiments != set(experiment_ids):
        raise ValueError("Verified source records and response-window tables disagree on experiment identity.")
    experiment_titles = {
        experiment_id: experiment_display_title_from_config(
            bundle_root / "sources" / experiment_id / "config.yaml",
            expected_experiment_id=experiment_id,
        )
        for experiment_id in experiment_ids
    }
    experiment_options = experiment_selector_options(experiment_titles)
    return experiment_ids, experiment_options, experiment_titles


@app.cell
def _(experiment_options, mo):
    experiment = mo.ui.dropdown(
        options=experiment_options,
        value=next(iter(experiment_options)),
        label="Experiment",
        searchable=True,
        full_width=True,
    )
    return (experiment,)


@app.cell
def _(bundle_manifest, display_contract, mo):
    _state_order = tuple(str(value) for value in bundle_manifest["state_order"])
    _state_labels = display_contract["state_labels"]
    condition_options = {f"{_state_labels[state]} ({state})": state for state in _state_order}
    condition = mo.ui.dropdown(
        options=condition_options,
        value=next(iter(condition_options)),
        label="Condition",
        full_width=True,
    )
    return condition, condition_options


@app.cell
def _(
    bundle_manifest,
    design_rows,
    display_contract,
    experiment_ids,
    experiment_titles,
    response_window_review_collection,
):
    _primary_nonreference = design_rows.loc[
        design_rows["reduction_role"].astype(str).eq("primary")
        & ~design_rows["is_reference"]
    ]
    if _primary_nonreference.empty:
        review_collection = None
        multi_design_options = {}
    else:
        review_collection = response_window_review_collection(
            design_rows,
            experiment_ids=experiment_ids,
            experiment_titles=experiment_titles,
            review_collection_id=str(bundle_manifest["request_id"]),
            review_collection_label=str(display_contract["study_label"]),
        )
        multi_design_options = review_collection.multi_experiment_entity_options()
    return multi_design_options, review_collection


@app.cell
def _(
    mo,
    multi_design_options,
    retained_review_option_key,
    review_design_get,
    review_design_set,
):
    multi_design = (
        mo.ui.dropdown(
            options=multi_design_options,
            value=retained_review_option_key(
                multi_design_options,
                preferred_id=review_design_get(),
            ),
            label="Reader design",
            searchable=True,
            on_change=review_design_set,
            full_width=True,
        )
        if multi_design_options
        else None
    )
    return (multi_design,)


@app.cell
def _(design_rows, experiment, mo, retained_review_selection, review_design_get, review_design_set):
    design_options = sorted(
        design_rows.loc[
            design_rows["experiment_id"].astype(str).eq(experiment.value),
            "design_id",
        ].astype(str).unique().tolist()
    )
    if not design_options:
        raise ValueError(f"Experiment {experiment.value!r} contains no Reader designs.")
    design = mo.ui.dropdown(
        options=design_options,
        value=retained_review_selection(
            design_options,
            preferred_id=review_design_get(),
        ),
        label="Reader design",
        searchable=True,
        on_change=review_design_set,
        full_width=True,
    )
    return design, design_options


@app.cell
def _(VIEW_LABELS, mo, multi_design_options):
    available_view_labels = {
        label: view_id
        for label, view_id in VIEW_LABELS.items()
        if view_id != "multi_experiment_evidence" or multi_design_options
    }
    view = mo.ui.dropdown(
        options=available_view_labels,
        value=next(iter(available_view_labels)),
        label="Review view",
        full_width=True,
    )
    return available_view_labels, view


@app.cell
def _(display_contract, review_view_spec, view):
    view_contract = review_view_spec(view.value, display=display_contract)
    return (view_contract,)


@app.cell
def _(
    design,
    experiment,
    experiment_selector_options,
    multi_design,
    review_collection,
    view_contract,
):
    if view_contract.selection_scope == "multi_experiment_design":
        active_design_id = multi_design.value
        _experiments = review_collection.experiments_for_entity(active_design_id)
        _raw_labels = {item.experiment_id: item.display_title for item in _experiments}
        _option_labels = {
            experiment_id: label
            for label, experiment_id in experiment_selector_options(_raw_labels).items()
        }
        active_experiment_labels = {
            item.experiment_id: _option_labels[item.experiment_id] for item in _experiments
        }
        active_experiment_id = None
    elif view_contract.selection_scope == "review_collection":
        active_design_id = None
        active_experiment_id = None
        active_experiment_labels = None
    else:
        active_design_id = design.value
        active_experiment_id = experiment.value
        active_experiment_labels = None
    return active_design_id, active_experiment_id, active_experiment_labels


@app.cell
def _(
    active_design_id,
    active_experiment_id,
    active_experiment_labels,
    common_cross_experiment_reductions,
    design_rows,
    mo,
    retained_review_option_key,
    response_summary_options,
    review_summary_get,
    review_summary_set,
    view_contract,
):
    if view_contract.selection_scope == "multi_experiment_design":
        available_reductions = common_cross_experiment_reductions(
            design_rows,
            design_id=active_design_id,
            experiment_ids=tuple(active_experiment_labels),
        )
    elif view_contract.selection_scope == "review_collection":
        available_reductions = design_rows.copy()
    else:
        available_reductions = design_rows.loc[
            design_rows["experiment_id"].astype(str).eq(active_experiment_id)
            & design_rows["design_id"].astype(str).eq(active_design_id)
        ].copy()
    reduction_options = response_summary_options(available_reductions)
    if not reduction_options:
        raise ValueError("Active review selection contains no response-window summaries.")
    reduction = mo.ui.dropdown(
        options=reduction_options,
        value=retained_review_option_key(
            reduction_options,
            preferred_id=review_summary_get(),
        ),
        label="Response summary",
        on_change=review_summary_set,
        full_width=True,
    )
    return available_reductions, reduction, reduction_options


@app.cell
def _(
    active_design_id,
    condition,
    design,
    display_contract,
    experiment,
    experiment_titles,
    mo,
    multi_design,
    reduction,
    view,
    view_contract,
):
    _channels = display_contract["channels"]
    _study_title = str(display_contract["study_label"])
    if view_contract.selection_scope == "experiment_design":
        _title = experiment_titles[experiment.value]
        _context_line = f"**Review collection:** {_study_title}"
    elif view_contract.selection_scope == "multi_experiment_design":
        _title = _study_title
        _context_line = f"**Across experiments:** `{active_design_id}`"
    else:
        _title = _study_title
        _context_line = "**Review collection overview**"
    introduction = mo.md(
        f"""
        # {_title}
        {_context_line}

        Connect growth and fluorescence after {str(display_contract['event_label']).lower()} to four
        `{_channels['response_ratio']}` responses and four `{_channels['reference_design_id']}`-relative
        `{_channels['magnitude_ratio']}` fluorescence values. Exact Reader identities remain available below.
        """
    )
    control_items = [view]
    control_widths = [1.15]
    if view_contract.selection_scope == "experiment_design":
        control_items.extend([experiment, design])
        control_widths.extend([1.55, 1.05])
    elif view_contract.selection_scope == "multi_experiment_design":
        control_items.append(multi_design)
        control_widths.append(1.2)
    if view_contract.condition_mode == "selected":
        control_items.append(condition)
        control_widths.append(1.2)
    if view_contract.reduction_mode == "selected":
        control_items.append(reduction)
        control_widths.append(1.4)
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
    active_design_id,
    active_experiment_id,
    active_experiment_labels,
    bundle_manifest,
    condition,
    cross_experiment_design_rows,
    design_rows,
    display_contract,
    event_rows,
    experiment_titles,
    measured_response_example_rows,
    mo,
    plt,
    reduction,
    render_review_figure,
    selected_handoff_row,
    trace_rows,
    view,
    view_contract,
    well_rows,
):
    active_reduction = reduction.value
    if view_contract.selection_scope == "multi_experiment_design":
        selected = cross_experiment_design_rows(
            design_rows,
            design_id=active_design_id,
            reduction_id=active_reduction,
        )
    elif view_contract.selection_scope == "review_collection":
        selected = measured_response_example_rows(
            design_rows,
            display=display_contract,
            reduction_id=active_reduction,
        )
    elif view_contract.reduction_mode == "all":
        selected = design_rows.loc[
            design_rows["experiment_id"].astype(str).eq(active_experiment_id)
            & design_rows["design_id"].astype(str).eq(active_design_id)
        ].copy()
    else:
        selected = selected_handoff_row(
            design_rows,
            experiment_id=active_experiment_id,
            design_id=active_design_id,
            reduction_id=active_reduction,
        )
    figure = render_review_figure(
        view_id=view.value,
        experiment_id=active_experiment_id,
        design_id=active_design_id,
        reduction_id=active_reduction,
        state=condition.value if view_contract.condition_mode == "selected" else None,
        experiment_labels=active_experiment_labels,
        designs=design_rows,
        wells=well_rows,
        traces=trace_rows,
        events=event_rows,
        display=display_contract,
    )
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
    if view_contract.selection_scope == "review_collection":
        evidence_columns = ["example_label", "example_role", *evidence_columns]
    _accordion_items = {
        "Handoff evidence": mo.ui.dataframe(selected.loc[:, evidence_columns]),
        "Evidence interpretation": mo.md(
            f"""
            **Premise:** {view_contract.premise}

            **Decision value:** {view_contract.decision_value}

            **How to read it:** {view_contract.interpretation}

            **Figure description:** {view_contract.alt_text}

            **Limit:** {view_contract.non_claim_boundary}
            """
        ),
        "Bundle provenance": mo.md(
            f"""
            - **Request:** `{bundle_manifest['request_id']}`
            - **Bundle schema:** `{bundle_manifest['schema_version']}`
            - **Primary response summary:** `{bundle_manifest['primary_reduction_id']}`
            - **Verified source snapshots:** {len(bundle_manifest['source_records'])}
            """
        ),
    }
    if view_contract.selection_scope == "multi_experiment_design":
        _coverage = selected.loc[:, ["experiment_id", "design_id", "reduction_id"]].copy()
        _coverage.insert(
            1,
            "experiment_title",
            _coverage["experiment_id"].astype(str).map(experiment_titles),
        )
        _accordion_items = {
            "Experiment coverage": mo.ui.dataframe(_coverage),
            **_accordion_items,
        }
    evidence = mo.accordion(_accordion_items, lazy=True)
    figure_view = mo.as_html(figure).style({"max-width": "100%", "overflow": "hidden"})
    plt.close(figure)
    mo.vstack([figure_view, evidence], gap=0.75)
    return evidence, evidence_columns, figure_view, selected


if __name__ == "__main__":
    app.run()
'''


__all__ = ["write_review_notebook"]
