# Running notebooks

Once you run a pipeline you can generate [marimo notebooks](https://marimo.io/) to explore outputs.

### Contents

1. [General usage](#general-usage)
2. [Using reader templates](#using-reader-templates)

---

### General usage

In general there are two ways to use marimo:

1. Install marimo into the project

    ```bash
    uv sync --locked --group notebooks
    uv run marimo edit outputs/notebooks/foo.py
    ```

    This runs marimo inside your project environment, so it can import **reader** and anything in `uv.lock`.

2. Sandboxed / self-contained marimo notebooks (inline dependencies)

    Marimo can manage per-notebook sandbox environments using inline metadata. This is great for shareable notebooks.

    Create/edit a sandbox notebook (marimo installed temporarily via uvx).

    ```bash
    uvx marimo edit --sandbox outputs/notebooks/sandbox_example.py
    ```

    Run a sandbox notebook as a script.

    ```bash
    uv run outputs/notebooks/sandbox_example.py
    ```

3. Make the sandbox notebook use your local reader repo in editable mode.

    From the repo root:

    ```bash
    uv add --script outputs/notebooks/sandbox_example.py . --editable
    ```

    This writes inline metadata into the notebook so its sandbox can install reader from your local checkout in editable mode.

4. Add/remove sandbox dependencies (only affects the notebook file)

    ```bash
    uv add    --script outputs/notebooks/sandbox_example.py numpy
    uv remove --script outputs/notebooks/sandbox_example.py numpy
    ```

> Note: You can also run claude code/codex in the terminal and ask it to edit a marimo notebook on your behalf. Make sure that you run your notebook with the watch flag turned on, like `marimo edit --watch notebook.py`, to see updates appear live whenever an agent makes a change.

---

### Using reader templates

Templates let you scaffold a ready-to-run marimo notebook that’s already wired to your experiment outputs.
Use `uv run reader notebook` for broad exploration across dataframe records.
By default, notebooks are written under `outputs/notebooks/`.

Scaffold a notebook (opens Marimo by default):

```bash
uv run reader notebook experiments/my_experiment/config.yaml
```

What the scaffolded notebook includes:

* dataframe record discovery via `outputs/manifests/records.json`
* a dataset dropdown labeled “Dataset (dataframe record)” (defaults to the most downstream step when possible)
* a canonical dataframe selection variable backed by the chosen parquet file (polars required to read parquet)
* a compact experiment overview with experiment id/title plus a `design_id` / `treatment` summary when those columns exist
* a dataset table explorer (`mo.ui.table`) driven by the dataset dropdown
* load-status messaging when no records exist yet or parquet loading fails

The default `notebook/eda`, `notebook/basic`, and `notebook/microplate` templates are intentionally minimal record explorers.
They do not currently scaffold ad-hoc plotting controls or Altair chart builders.
`plate_reader/retron_sponge_screen` instead defaults to `notebook/retron_sponge`, which adds an
experiment-scoped plot-portfolio review, transform ladder, and semantic-table walkthrough on top of the record explorer.
For cross-run retron library review, `notebook/retron_sponge_aggregate` is available as an explicit opt-in template
for generic review experiments that aggregate exported semantic tables from multiple source screens.

The dataset dropdown drives the canonical `df_active` variable.

See what’s available:

```bash
uv run reader notebook --list-templates
```

Notes:

* `uv run reader notebook` only scaffolds the notebook; it does not run the pipeline.
* `uv run reader notebook` launches Marimo with the active Python interpreter (e.g., `sys.executable -m marimo ...`), so running via `uv run` ensures the notebook deps are available.
* Use `--mode none` to scaffold without launching Marimo, or `--mode run` to launch a read-only app.
* Record discovery is catalog-first. If `outputs/manifests/records.json` is missing, the scaffolded notebook will show no datasets unless you regenerate records with `uv run reader run` or opt in with `uv run reader notebook --scan-records`.
* Common templates include `notebook/retron_sponge`, `notebook/retron_sponge_aggregate`, `notebook/eda`, `notebook/basic`, `notebook/microplate`, `notebook/cytometry`, and `notebook/sfxi_eda`.
* Template behavior is capability-driven:
  - plot filtering is only available for templates that declare plot-filter support
  - auto-pick chooses a template from declared default rules instead of hardcoded CLI branching
  - template applicability checks are declared on the template asset itself
* `notebook/sfxi_eda` requires SFXI-capable context declared through asset requirements: either an SFXI-tagged pipeline transform or compatible dataframe records.
* The SFXI template draws a red dashed induction marker on the time-series plot when an induction time can be inferred from dataframe records:
  - preferred: an explicit column like `induction_time_h` (or `induction_time`) in the tidy dataframe
  - fallback: Synergy H1 ingest columns (`sheet_index` + `time`), where the first time in the second sheet is treated as the induction time
  If neither is present, the induction marker is omitted.
* If the target notebook already exists, use `--force` (or `--refresh`) to overwrite it, or `--new` to create a second notebook with an automatic numeric suffix.
* If `--template` is omitted, reader uses the first configured `notebooks.specs` entry from `config.yaml` if provided; otherwise it auto-selects the first template whose declared default rule matches the workbench state.

See also: [SFXI vec8 in reader](../lib/sfxi_vec8_in_reader.md) for how the vec8 pipeline is computed and how the SFXI notebook template aligns with the code.
