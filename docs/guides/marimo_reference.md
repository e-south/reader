---
doc_id: reader-marimo-reference
surface: agent-reference
owner: reader-maintainers
last_verified: 2026-07-17
summary: Reader-specific Marimo authoring, performance, accessibility, and validation contract.
---

# Marimo reference for Reader notebooks

Use this page when editing a Reader-managed Marimo notebook or template. It
covers the repository-specific rules that are easy to miss. For the complete
Marimo component and API reference, use the
[official Marimo documentation](https://docs.marimo.io/).

For operator commands and canonical scaffolding, start with
[Running notebooks](./notebooks.md). For the files and records that notebooks
consume, use [Configuring Reader v8](../core/pipeline.md).

## Managed workflow

Prefer Reader's launcher because it resolves the experiment, generated
notebook path, cache directory, loopback server, and stale-session restart
policy:

```bash
uv run reader notebook CONFIG|DIR|INDEX --mode run --headless
```

Use Marimo directly only when the Reader launcher is not the owning surface:

```bash
uv run marimo check path/to/notebook.py
uv run marimo edit --watch path/to/notebook.py
```

Hand-authored notebooks belong in `experiments/<experiment>/notebooks/`.
`reader notebook` writes generated scaffolds under `outputs/notebooks/`.
Change a template or package helper when generated notebook behavior is wrong;
do not patch one generated output by hand.

## Reactive contract

Marimo builds a dependency graph from cell definitions and references.

- Define each global name in exactly one cell.
- Keep UI creation and `.value` access in separate cells.
- Do not create dependency cycles or use `global`.
- Prefix cell-local temporaries with `_`.
- Make a display object the final expression in its cell. Do not call
  `plt.show()`.
- Keep cells idempotent where practical. A reactive rerun should not append,
  overwrite, or mutate unrelated state silently.

When editing an existing notebook, change only the body of an `@app.cell`
function unless the defect is the cell declaration itself. Marimo owns the
function signature and return wiring.

```python
@app.cell
def _(mo):
    threshold = mo.ui.slider(0.0, 1.0, value=0.5, label="Threshold")
    threshold
    return (threshold,)


@app.cell
def _(threshold):
    selected_threshold = threshold.value
    return (selected_threshold,)
```

## Data and performance

Use the record catalog to select data. Do not scan arbitrary files when a
typed record exists.

- Read cataloged dataframes through `reader_workbench.api.read_dataframe` with the exact
  revision and revision digest returned by `reader_workbench.api.records`.
- Keep filtering, grouping, pivoting, and statistics in Polars when possible.
- Convert to Pandas only at a public plotting boundary that requires it, and
  convert a bounded or downsampled payload rather than the full record.
- Put reusable assay calculations in `reader_workbench.domains`, not in a large notebook
  cell.
- Gate expensive deterministic work behind its true dependencies. Display-only
  controls must not trigger data preparation or statistics again.
- Use a bounded cache only when a measured rerender cost justifies it. Cache
  keys must include every semantic input, and cached Matplotlib figures must be
  closed on eviction.

These rules matter for cytometry and aggregate reviews, where a compact
Parquet file can expand to hundreds of megabytes after eager materialization or
wide conversion.

Cytometry gating follows this contract before the notebook: the normal
`transform/cytometry_gating` step persists bounded summary and QC records, and
`plot/cytometry_diagnostic` downsamples only its display payload. Prefer those
records and the registered diagnostic over loading the full gated-event table
for routine review.

## Scientific visual descriptions

Every displayed or persisted figure needs a short description that lets a
reader understand the scientific job of the visual without inferring it from
the filename.

A useful description states:

1. what is compared
2. the measurement or transformation on each axis or panel
3. the grouping, control, or normalization that changes interpretation
4. the main limit, such as a missing state or a display-only projection

Keep the description descriptive, not interpretive. Do not claim a biological
effect that the plotted evidence does not establish.

Use public rendering metadata where it is available:

```python
mo.image(image, alt=figure_description, caption=figure_caption)
```

For Altair, set a concise chart `description` and keep readable titles, axis
labels, units, scales, legends, and tooltips. For Matplotlib, show the figure
beside a Markdown description or through a Reader component that carries the
same text. Protocol figure summaries are the canonical starting point; a
notebook may add experiment-specific context without changing the underlying
figure meaning.

## Progressive disclosure

Reader notebooks should present information in this order:

1. experiment identity, protocol, and pipeline scope
2. dataset and semantic selectors
3. primary scientific figures and concise descriptions
4. supporting tables, output paths, and provenance
5. raw records, diagnostics, and export controls

Use lazy accordions or tabs for supporting detail. Keep the primary result and
its interpretation boundary visible without opening every panel. Do not bury a
blocking validation error inside a collapsed section.

When a compact endpoint panel has source-declared replicate values, show those
values as neutral hollow points and the published aggregate as a short line.
Name the interval statistic explicitly. Do not use a bar as a redundant
encoding of an aggregate, combine independent uncertainty sources into one
error bar, or draw pseudo-replicates for a quantity defined from independent
sample aggregates.

Use `experiment.title` as the visible notebook title when it is authored. If it
is absent, Reader derives a deterministic display title from `experiment.id`;
the ID remains the machine identity and belongs in compact provenance detail,
not in the main heading. For an experiment review, prefer one concise purpose
sentence, one selector row, one primary figure viewport, and a small accordion
for handoff values, metadata, outputs, and raw records.

## Validation

After changing the canonical notebook scaffold:

```bash
uv run pytest -q src/reader_workbench/tests/notebooks/test_canonical_notebook.py
uv run pytest -q src/reader_workbench/tests/notebooks
uv run ruff check .
uv run ruff format . --check
```

The notebook suite renders the fixed scaffold to a temporary `.py` file and
runs `marimo check` when notebook dependencies are installed. It also
checks for duplicate globals and cell-shaped functions that are missing an
`@app.cell` decorator.

For a representative experiment, use the Reader launcher and verify the real
interactive path. A clean `marimo check` does not prove that a selector,
button, or chart works after a click.

## Common failures

- Missing producer for a cell input
  - Confirm the producing function has `@app.cell` and returns the global.
- Duplicate global or dependency cycle
  - Move temporary names behind `_`, split responsibilities, or remove the
    reverse dependency.
- Data reloads after a display-only change
  - Separate data preparation from rendering and narrow the producer's inputs.
- Notebook consumes stale or unverified tables
  - Resolve the typed record and use its verified load path instead of a nearby
    export file.
- Plot is visible but not self-contained
  - Add a concise scientific description, axis units, control or normalization
    context, and the relevant evidence limit.
- Browser behavior differs from script checks
  - Run the real launcher path and inspect the live reactive state; do not treat
    lint success as interactive verification.
