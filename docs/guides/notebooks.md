---
doc_id: reader-notebooks-guide
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-07-08
summary: Reader marimo notebook workflow, template selection, component ownership, and live-review checks.
---

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

For browser review without opening the editor, prefer:

```bash
uv run reader notebook experiments/my_experiment/config.yaml --mode run --headless
```

What the scaffolded notebook includes:

* dataframe record discovery via `outputs/manifests/records.json`
* a dataset dropdown labeled “Dataset (dataframe record)” (defaults to the most downstream step when possible)
* a canonical dataframe selection variable backed by the chosen parquet file (polars required to read parquet)
* a compact experiment overview with experiment id, protocol, pipeline steps, paths, and `design_id` / `treatment` vocabulary when those columns exist
* a progressive-disclosure deliverables panel for manifest-backed records, plots, exports, and generated notebooks
* a dataset table explorer (`mo.ui.table`) driven by the dataset dropdown
* load-status messaging when no records exist yet or parquet loading fails

The default `notebook/eda`, `notebook/basic`, and `notebook/microplate` templates are intentionally minimal record explorers.
They do not currently scaffold ad-hoc plotting controls or Altair chart builders.
`notebook/dual_reporter_triptych` is a neutral plate-reader review surface for dual-reporter assays. It renders
OD600 kinetics, YFP/CFP kinetics, and a YFP/CFP snapshot bar plot for one selected design without assuming SFXI
four-corner logic or vec8 export semantics.
`plate_reader/retron_sponge_screen` instead defaults to `notebook/retron_sponge`, which adds an
experiment-scoped plot-portfolio review, transform ladder, and semantic-table walkthrough on top of the record explorer.
For cross-run retron library review, `notebook/retron_sponge_aggregate` is available as an explicit opt-in template
for generic review experiments that aggregate exported semantic tables from multiple source screens.

Template selection is ordered and protocol-constrained:

1. explicit `--template`
2. first compiled notebook spec from `config.yaml`
3. bound protocol default

The selected template must be listed in the protocol's allowed notebook
templates. This keeps template choice semantic without letting template
capabilities silently override the experiment contract.

Generated notebooks keep shared review pieces in
`reader.workbench.notebooks.components`:

* `overview` owns frontmatter, path summaries, pipeline rows, and the
  design/treatment vocabulary table.
* `deliverables` owns manifest-backed records plus plot, export, and generated
  notebook file bundles.
* assay-specific templates can add domain review sections above or beside
  those panels, but should avoid duplicating component-owned tables.

The dataset dropdown drives the canonical dataframe selection used by the
record explorer and assay-specific sections.

See what’s available:

```bash
uv run reader notebook --list-templates
```

Notes:

* `uv run reader notebook` only scaffolds the notebook; it does not run the pipeline.
* `uv run reader notebook` launches Marimo with the active Python interpreter, so running via `uv run` ensures the notebook deps are available.
* `reader notebook` manages Marimo runtime state under `.cache/marimo/` in the repo. It uses clean repo-local XDG and Matplotlib cache directories instead of leaking into user-global Marimo state.
* `reader notebook` reuses an existing reader-managed session for the same notebook only when the notebook file and reader runtime fingerprint still match. If either has drifted, it restarts the stale session instead of silently attaching to it.
* It also prunes older reader-managed sessions for the same experiment and launch mode before starting a new one.
* Use `--mode none` to scaffold without launching Marimo, `--mode run` to launch a read-only app, and `--headless` when an agent or browser automation should attach to the printed loopback URL.
* Use `--port <n>` only when you need a fixed loopback port. Otherwise let `reader` choose a clean port starting at `2718`.
* For agent review, the low-friction path is:
  - `uv run reader notebook <config> --mode run --headless`
  - open the printed URL in Chrome MCP
  - or run `uv run marimo check <notebook.py>` for a static validation pass
* Static HTML export is useful as an execution/shareability smoke check, but it is not an interaction check. Validate dropdowns, sliders, export buttons, and chart rerenders from a live `marimo run` app.
* Record discovery is catalog-first. If `outputs/manifests/records.json` is missing, the scaffolded notebook will show no datasets unless you regenerate records with `uv run reader run` or opt in with `uv run reader notebook --scan-records`.
* Common templates include `notebook/retron_sponge`, `notebook/retron_sponge_aggregate`, `notebook/eda`, `notebook/basic`, `notebook/microplate`, `notebook/dual_reporter_triptych`, `notebook/cytometry`, and `notebook/sfxi_eda`.
* Template behavior is contract-driven:
  - plot filtering is only available for templates that declare plot-filter support
  - default selection uses the compiled notebook spec or the protocol default instead of hardcoded CLI branching
  - template applicability checks are declared on the template asset itself
* `notebook/sfxi_eda` requires SFXI-capable context declared through asset requirements: either an SFXI-tagged pipeline transform or compatible dataframe records.
* `notebook/sfxi_eda` reuses the neutral dual-reporter triptych for visualization, then layers SFXI-specific vec8 recomputation, reference anchoring, and XLSX/JSON export on top.
* The SFXI template draws a red dashed induction marker on the time-series plot when an induction time can be inferred from dataframe records:
  - preferred: an explicit column like `induction_time_h` (or `induction_time`) in the tidy dataframe
  - fallback: Synergy H1 ingest columns (`sheet_index` + `time`), where the first time in the second sheet is treated as the induction time
  If neither is present, the induction marker is omitted.
* If the target notebook already exists, use `--force` (or `--refresh`) to overwrite it, or `--new` to create a second notebook with an automatic numeric suffix.
* If `--template` is omitted, reader uses the first configured `notebooks.specs` entry from `config.yaml` if provided; otherwise it uses the bound protocol default.

See also: [SFXI vec8 in reader](../lib/sfxi_vec8_in_reader.md) for how the vec8 pipeline is computed and how the SFXI notebook template aligns with the code.
