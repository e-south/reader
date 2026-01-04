# Running notebooks

Once you run a pipeline you can generate [marimo notebooks](https://marimo.io/) to explore outputs.

### Contents

1. [General usage](#general-usage)
2. [Using reader presets](#using-reader-presets)
3. [Plot-focused notebooks](#plot-focused-notebooks)

---

### General usage

In general there are two ways to use marimo:

1. Install marimo into the project

    ```bash
    uv sync --locked --group notebooks
    uv run marimo edit notebooks/foo.py
    ```

    This runs marimo inside your project environment, so it can import **reader** and anything in `uv.lock`.

2. Sandboxed / self-contained marimo notebooks (inline dependencies)

    Marimo can manage per-notebook sandbox environments using inline metadata. This is great for shareable notebooks.

    Create/edit a sandbox notebook (marimo installed temporarily via uvx).

    ```bash
    uvx marimo edit --sandbox notebooks/sandbox_example.py
    ```

    Run a sandbox notebook as a script.

    ```bash
    uv run notebooks/sandbox_example.py
    ```

3. Make the sandbox notebook use your local reader repo in editable mode.

    From the repo root:

    ```bash
    uv add --script notebooks/sandbox_example.py . --editable
    ```

    This writes inline metadata into the notebook so its sandbox can install reader from your local checkout in editable mode.

4. Add/remove sandbox dependencies (only affects the notebook file)

    ```bash
    uv add    --script notebooks/sandbox_example.py numpy
    uv remove --script notebooks/sandbox_example.py numpy
    ```

> Note: You can also run claude code/codex in the terminal and ask it to edit a marimo notebook on your behalf. Make sure that you run your notebook with the watch flag turned on, like `marimo edit --watch notebook.py`, to see updates appear live whenever an agent makes a change.

---

### Using reader presets

Presets let you scaffold a ready-to-run marimo notebook that’s already wired to your experiment outputs.
Use `reader notebook` for broad exploration across artifacts, plots, and exports. Use `reader plot --mode notebook` when you want a plot-focused notebook for specific plot spec ids.

Scaffold a notebook (and open it):

```bash
reader notebook experiments/my_experiment/config.yaml --preset notebook/basic --edit
```

What the scaffolded notebook includes:

* artifact discovery via `outputs/manifest.json`
* plot/export listing via `outputs/plots_manifest.json` and `outputs/exports_manifest.json` (optional if you never saved outputs)
* a summary of configured plot specs from `config.yaml`
* a view of available plot modules under `reader.plugins.plot`
* a lightweight metadata overview (e.g., design/treatment keys + time coverage) for a selected artifact
* fast EDA helpers: reactive Altair plots and DuckDB-backed parquet queries
* correct path resolution using `paths.outputs`, `paths.plots`, and `paths.exports`

See what’s available:

```bash
reader notebook --list-presets
```

Notes:

* `reader notebook` only scaffolds the notebook; it does not run the pipeline.
* Use `--edit` to open the notebook immediately (avoids copy/paste of the path).
* Common presets include `notebook/basic`, `notebook/microplate` (time-series + snapshots), and `notebook/cytometry` (population distributions).
* If the target notebook already exists, use `--force` (or `--refresh`) to overwrite it, or `--new <FILE>` to create a second notebook.
* The EDA notebooks rely on Altair for quick plots; install the notebooks group (`uv sync --locked --group notebooks`) or add it directly (`uv add altair`) if you see an Altair missing error.

---

### Plot-focused notebooks

You can scaffold a plot-specific notebook without running plot plugins directly:

```bash
reader plot experiments/my_experiment/config.yaml --mode notebook --only plot_time_series --edit
```

This generates a notebook that can execute the selected plot spec(s) interactively inside marimo.
It reads plot specs from the config and artifacts from `outputs/manifest.json` — it does not require saved plots.

---

@e-south
