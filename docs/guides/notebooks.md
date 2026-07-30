---
doc_id: reader-notebooks-guide
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-07-29
summary: Operate Reader's canonical record-driven Marimo workbench.
---

# Running notebooks

Reader generates one fixed scaffold, packaged as `notebook/eda`. It is a read-only viewport
over an experiment's verified record catalog. Assay computation, plots, and
exports remain normal protocol-owned steps.

## Canonical workflow

Run and verify the experiment before opening its notebook:

```bash
uv run reader validate experiments/my_experiment/config.yaml
uv run reader run experiments/my_experiment/config.yaml
uv run reader plot experiments/my_experiment/config.yaml
uv run reader export experiments/my_experiment/config.yaml
uv run reader verify experiments/my_experiment/config.yaml
uv run reader notebook experiments/my_experiment/config.yaml --mode run --headless
```

The generated notebook contains:

- a compact experiment and pipeline overview;
- one dropdown for dataframe records and registered plot, export, or other
  file-bundle records;
- one viewport for the selected table, image, or PDF;
- one lazy accordion for metadata, catalog summaries, and readiness issues.

The notebook reads only through `reader_workbench.api.records`, `reader_workbench.api.verify`,
`reader_workbench.api.read_dataframe`, and `reader_workbench.api.read_artifact`. Each preview is
bound to the selected record revision and revision digest. Reader verifies the
exact dataframe bytes or file bytes before rendering them.

The notebook does not scan output directories, infer artifact ownership, or
publish records. A generated notebook is operator scaffolding, not a
scientific-record producer.

## Adding a review surface

Do not add an assay-specific notebook lifecycle. Put reusable behavior at its
owning layer:

- domain math and rendering under `reader_workbench.domains`;
- thin executable adapters under `reader_workbench.plugins`;
- assay composition and defaults under `reader_workbench.protocols`;
- persisted dataframes and files in the normal `RecordStore` lifecycle.

The canonical notebook discovers new plot and export records automatically.
Study labels, objectives, and interpretation stay with the consuming study;
Reader may render authored labels but does not invent their meaning.

Examples include the normal `dual_reporter_triptych`,
`response_window_diagnostic`, `sfxi_diagnostic`, and
`cytometry_diagnostic` plot outputs. Each consumes persisted records and can be
selected in the same viewport after `reader plot`.

## Scaffolding and launch modes

Scaffold without installing Marimo:

```bash
uv run reader notebook experiments/my_experiment/config.yaml --mode none
```

Reader does not publish a `notebooks` extra while the released Marimo and
PyMdown constraints lack a dependency resolution Reader can safely advertise.
For `edit` or `run` modes, use a separately managed and audited environment
that provides Marimo, Altair, and DuckDB, then invoke the `reader` command from
that environment. Do not bypass dependency conflicts with an installer
override.

Notebooks are written beneath the owning experiment's configured
`outputs/notebooks/` directory. Use `--name EDA_custom.py` to choose a filename,
`--refresh` to regenerate it, or `--new` to create a numbered sibling. Do not
hand-edit generated scaffolds; change the shared scaffold source or component and
regenerate. The packaged scaffold has one stable identity; filenames are not
template choices.

Reader manages loopback Marimo sessions and repo-local runtime caches. It
reuses a session only when the notebook and Reader runtime fingerprints still
match. Use `--port` only when a fixed local port is necessary.

For a static check:

```bash
uv run marimo check outputs/notebooks/EDA_YYYYMMDD.py
```

A static HTML export is useful for execution and shareability checks, but use a
live `marimo run` session to verify interactive selection and rendering. See
the [Marimo reference](./marimo_reference.md) for component and performance
rules.

## Aggregate experiments

Cross-experiment review remains an experiment, not a notebook exception. For
example, an SFXI vec8 collection declares exact Reader record resources and
runs the ordinary lifecycle:

```bash
uv run reader init experiments/vec8_collection \
  --protocol logic/sfxi_vec8_collection \
  --title "SFXI vec8 collection"
uv run reader validate experiments/vec8_collection
uv run reader run experiments/vec8_collection
uv run reader plot experiments/vec8_collection
uv run reader export experiments/vec8_collection
uv run reader verify experiments/vec8_collection
```

See [SFXI plot surfaces](../lib/sfxi/plots.md) and
[plate-reader metric outputs](../lib/plate_reader/metric_outputs.md) for the
record and plot contracts.
