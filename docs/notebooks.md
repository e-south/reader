# Running notebooks

Once you run a pipeline you can generate [marimo notebooks](https://marimo.io/) to read and explore outputs.

### Contents

1. [General usage](#general-usage)
2. [Using `reader` presets](#using-reader-presets)

---

### General usage

In general there are two ways to use marimo:

1. Install marimo into the project

    ```bash
    uv sync --locked --group notebooks
    uv run marimo edit notebooks/foo.py
    ```

    This runs marimo inside your project environment, so it can import **dnadesign** and anything in `uv.lock`.

2. Sandboxed / self-contained marimo notebooks (inline dependencies)

    Marimo can manage per-notebook sandbox environments using inline metadata. This is great for sharable notebooks.

    Create/edit a sandbox notebook (marimo installed temporarily via uvx).

    ```bash
    uvx marimo edit --sandbox notebooks/sandbox_example.py
    ```

    Run a sandbox notebook as a script.

    ```bash
    uv run notebooks/sandbox_example.py
    ```
    
3. Make the sandbox notebook use your local dnadesign repo in editable mode.

    From the repo root:

    ```bash
    uv add --script notebooks/sandbox_example.py . --editable
    ```

    This writes inline metadata into the notebook so its sandbox can install dnadesign from your local checkout in editable mode.

4. Add/remove sandbox dependencies (only affects the notebook file)

    ```bash
    uv add    --script notebooks/sandbox_example.py numpy
    uv remove --script notebooks/sandbox_example.py numpy
    ```

> Note: You can also run claude code/codex in the terminal and ask it to edit a marimo notebook on your behalf. Make sure that you run your notebook with the watch flag turned on, like marimo edit --watch notebook.py, to see updates appear live whenever agents makes a change.

---

### Using reader presets

Presets let you scaffold a ready-to-run marimo notebook that’s already wired to your experiment outputs.

Scaffold a notebook:

```bash
reader explore experiments/my_experiment/config.yaml --preset eda/basic
```

What the scaffolded notebook includes:

* artifact discovery via `outputs/manifest.json`
* deliverable listing via `outputs/deliverables_manifest.json`
* a summary of configured plot steps from `config.yaml`, grouped by plot type
* a view of available plot modules under `reader.plugins.plot`
* a lightweight metadata overview (e.g., design/treatment keys + time coverage) for a selected artifact
* fast EDA helpers: reactive Altair plots and DuckDB-backed parquet queries
* correct path resolution using `experiment.outputs` and `experiment.plots_dir` (so custom output layouts work)

See what’s available:

```bash
reader explore --list-presets
```

Notes:

* `reader explore` only scaffolds the notebook; it does not run the pipeline.
* Common presets include `eda/basic`, `eda/microplate` (time-series + snapshots), and `eda/cytometry` (population distributions).



---

@e-south
