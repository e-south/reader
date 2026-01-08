# Running notebooks

Once you run a pipeline you can generate [marimo notebooks](https://marimo.io/) to explore outputs.

### Contents

1. [General usage](#general-usage)
2. [Using reader presets](#using-reader-presets)

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

### Using reader presets

Presets let you scaffold a ready-to-run marimo notebook that’s already wired to your experiment outputs.
Use `reader notebook` for broad exploration across artifact dataframes.
By default, notebooks are written under `outputs/notebooks/`.

Scaffold a notebook (opens Marimo by default):

```bash
uv run reader notebook experiments/my_experiment/config.yaml
```

What the scaffolded notebook includes:

* artifact discovery via `outputs/manifests/manifest.json`; if the manifest is missing or unreadable, the notebook
  will warn and scan `outputs/artifacts/**/df.parquet` to populate the dataset list
* a dataset dropdown labeled “Dataset (artifact df.parquet)” (defaults to the most downstream step when possible)
* a canonical `df_active` variable populated from the selected artifact (polars required to read parquet)
* dataset status (backend, rows/columns, parquet path)
* a dataset table explorer (`mo.ui.table`) driven by the dataset dropdown
* a top header with experiment id/title, resolved outputs path, and a conditions/treatments table
* a design + treatment summary panel (`design_id` / `treatment` if present)
* an ad-hoc EDA panel with x/y/hue/groupby dropdowns and Altair plotting

The dataset dropdown drives the canonical `df_active` variable.

See what’s available:

```bash
reader notebook --list-presets
```

Notes:

* `reader notebook` only scaffolds the notebook; it does not run the pipeline.
* `reader notebook` launches Marimo with the active Python interpreter (e.g., `sys.executable -m marimo ...`), so running via `uv run` ensures the notebook deps are available.
* Use `--mode none` to scaffold without launching Marimo, or `--mode run` to launch a read-only app.
* Common presets include `notebook/eda`, `notebook/basic`, `notebook/microplate`, `notebook/cytometry`, and `notebook/sfxi_eda` (SFXI vec8 builder scaffold; requires a `transform/sfxi` step or existing SFXI artifacts).
* The SFXI preset draws a red dashed induction marker on the time-series plot when an induction time can be inferred from artifacts:
  - preferred: an explicit column like `induction_time_h` (or `induction_time`) in the tidy dataframe
  - fallback: Synergy H1 ingest columns (`sheet_index` + `time`), where the first time in the second sheet is treated as the induction time
  If neither is present, the induction marker is omitted.
* If the target notebook already exists, use `--force` (or `--refresh`) to overwrite it, or `--new <FILE>` to create a second notebook.
* If `--preset` is omitted, reader uses `notebook.preset` from `config.yaml` if provided; otherwise it auto-selects `notebook/eda` when plots exist, or `notebook/basic` when they don't (both presets currently scaffold the same minimal notebook).

See also: [SFXI vec8 in reader](sfxi_vec8_in_reader.md) for how the vec8 pipeline is computed and how the SFXI notebook preset aligns with the code.

---

@e-south
