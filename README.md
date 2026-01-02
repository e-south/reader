[![CI](https://github.com/e-south/reader/actions/workflows/ci.yaml/badge.svg)](https://github.com/e-south/reader/actions/workflows/ci.yaml)

## reader

**reader** is a workbench for experimental data, where the unit of work is an **experimental directory**: you put raw inputs there, keep notebooks next to them, and write (`outputs/`) in the same place. **reader** includes a plugin-based pipeline runner (see [`docs/pipeline.md`](./docs/pipeline.md) and [`docs/plugins.md`](./docs/plugins.md)), but more broadly these workspaces are where you can iterate on one experiment with a mix of:

- **lightweight utilities** (`ls`, `run`, …)
- **repeatable steps** (via `config.yaml` + CLI)
- **exploratory work** (via marimo notebooks)

### Contents

- [Documentation](#documentation)
- [Repo layout](#repo-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI workbench commands](#cli-workbench-commands)
- [Running notebooks](#running-notebooks)
- [Maintaining dependencies](#maintaining-dependencies)
- [Upgrading dependencies](#upgrading-dependencies)

---

### Documentation

- [Pipeline config + reports](./docs/pipeline.md)
- [Plugin development](./docs/plugins.md)

---

### Repo layout

**reader** is basically two things: a place for experiments, and a library/CLI that helps you analyze them.

```bash
reader/
├─ experiments/             # experiment workbench (configs + data + results)
│  └─ exp_001/
│     ├─ config.yaml        # pipeline spec (ingest→merge→transform + optional reports)
│     ├─ inputs/            # instrument exports (preferred; raw_data/ or raw/ also scanned by default)
│     ├─ notebooks/         # optional: marimo notebooks for this experiment
│     └─ outputs/           # generated: artifacts/, plots/, exports/, manifest.json, report_manifest.json, reader.log
│
└─ src/
   └─ reader/               # optional: library for running config.yaml-driven workflows across experiments
      ├─ core/              # shared commands: run/explain/validate/ls/artifacts
      ├─ io/                # implement an instrument parser once (raw → tidy), reuse it across experiments
      ├─ plugins/           # thin adapters that expose io/lib operations to config.yaml steps
      │  ├─ ingest/         # raw → tidy artifacts (canonical tables)
      │  ├─ merge/          # tidy + metadata/table joins
      │  ├─ transform/      # reusable cleanups + derived columns/channels
      │  ├─ plot/           # optional shared plotting steps
      │  └─ validator/      # optional schema gates / normalizers (coercion, shape checks)
      ├─ lib/               # reusable domain helpers (imported by plugins + notebooks)
      └─ tests/
```

---

### Installation

This repo is managed with [**uv**](https://docs.astral.sh/uv/):

- `pyproject.toml` declares dependencies (runtime + optional extras).
- `uv.lock` is the fully pinned dependency graph.
- `.venv/` is the project virtual environment.

**Two key commands:**

- `uv sync` installs everything from the lockfile into `.venv`.
- `uv run <cmd>` runs commands inside the project environment without requiring `source .venv/bin/activate`.

 1. Install uv. Below is for macOS/Linux (for other OSs see [here](https://docs.astral.sh/uv/getting-started/installation/)).

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # ensure your uv bin dir is on PATH
    ```

2. Clone repo

    ```bash
    git clone https://github.com/e-south/reader.git
    cd reader
    ```

3. Create/sync the environment from the committed lockfile:

      ```bash
      uv sync --locked
      ```

4. Sanity checks:

      ```bash
      uv run python -c "import reader, pandas, pyarrow; print('ok')"
      ```

5. Dev tooling is opt-in via a dependency group.

    ```bash
    uv sync --locked --group dev --group notebooks
    uv run ruff --version
    uv run ruff check .
    uv run pytest -q
    ```

##### This project defines console scripts, which you can run via:

Option A: no `.venv` activation — use `uv run`

```bash
uv run reader --help
uv run reader ls
```

Option B: traditional — activate `.venv`

```bash
source .venv/bin/activate
reader --help
reader ls
deactivate
```

---

#### Quickstart

##### 1. If you have a template, copy it. Otherwise create a folder manually.

```bash
mkdir -p experiments/my_experiment/{inputs,notebooks,outputs}
```

Drop experimental data into `inputs/` (or `raw_data/` / `raw/`).

##### 2. Run CLI steps (optional, repeatable)

If the experiment has a `config.yaml`:
```bash
uv run reader explain experiments/my_experiment/config.yaml
uv run reader run     experiments/my_experiment/config.yaml   # runs pipeline + reports
```

---

#### CLI workbench commands

The CLI is a set of helpers for the workspace.
Commands assume `uv run reader` in this repo (or `reader` if installed).

```bash
reader ls --root experiments
reader demo
reader plugins
reader steps <CONFIG or INDEX>
reader artifacts <CONFIG or INDEX>
```

Common workflow helpers:

- `reader ls [--root DIR]` — list experiments (finds `**/config.yaml`).
- `reader demo` — quick walkthrough of common commands.
- `reader explain CONFIG|DIR|INDEX` — show the plan (what would run).
- `reader validate CONFIG|DIR|INDEX` — validate config + plugin configs (no data I/O).
- `reader check-inputs CONFIG|DIR|INDEX` — verify file inputs declared via `reads: file:` exist.
- `reader run CONFIG|DIR|INDEX [--step STEP] [--resume-from ID] [--until ID]` — execute the pipeline (reports run by default).
- `reader report CONFIG|DIR|INDEX [--list]` — render report steps (plots/exports) using existing artifacts.
- `reader run-step STEP --config CONFIG|DIR|INDEX` — run exactly one step using existing artifacts.
- `reader artifacts CONFIG|DIR|INDEX` — list the latest artifact locations (manifest-backed).
- `reader plugins` — show discovered plugins (built-ins + entry points).
- `reader presets [NAME]` — list preset bundles (or inspect one).
- `reader explore CONFIG|DIR|INDEX` — scaffold a marimo notebook under `notebooks/` (no execution).

**Note:** `ls` only lists experiments that have a `config.yaml`. Notebooks can exist without configs,
but then they won’t be discoverable via ls.

---

#### Running notebooks

There are two practical modes:

##### 0. Scaffold an experiment notebook (recommended)

```bash
reader explore path/to/experiment
uv sync --locked --group notebooks
uv run marimo edit path/to/experiment/notebooks/eda.py
```

The scaffolded notebook uses DuckDB to query parquet artifacts efficiently.
It includes a small “Quick plot” widget (line/scatter/hist) and a **Tidy explorer**
with channel/time filters, a data explorer, and a mean time-series view.

If you only want PDFs, use:

```bash
reader report path/to/experiment
```

If you use flow cytometry plugins, include the extra group:

```bash
uv sync --locked --group notebooks --group cytometry
```

##### 1. Install marimo into the project

```bash
uv sync --locked --group notebooks
uv run marimo edit notebook.py
```

This runs marimo inside your project environment, so it can import `reader` and anything in `uv.lock`.

##### 2. Optional: sandboxed marimo notebooks (inline dependencies)

Marimo can manage per-notebook environments using inline metadata (useful for sharing).

Create/edit a sandbox notebook:

```bash
uvx marimo edit --sandbox notebook.py
```

Add your local reader checkout to the sandbox:

```bash
uv add --script notebook.py . --editable
```

Add/remove sandbox dependencies (only affects the notebook file):

```bash
uv add    --script notebook.py numpy
uv remove --script notebook.py numpy
```

---

#### Maintaining dependencies

If you want to change dependencies, prefer `uv add` / `uv remove`:

- Add a runtime dependency:

  ```bash
  uv add <package>
  ```

- Add to a dependency group:

  ```bash
  uv add --group dev <package>
  ```

- Remove:

  ```bash
  uv remove <package>
  ```

Then commit `pyproject.toml` + `uv.lock`.

If you edit `pyproject.toml` by hand, regenerate `uv.lock`:

```bash
uv lock
```

New users should then run:

```bash
uv sync --locked
```

---

#### Upgrading dependencies

`dnadesign` is pulled from GitHub and pinned to a specific commit in `uv.lock`. If `dnadesign/main` changes, a plain `uv sync` will keep using the locked commit until you explicitly upgrade it (see [Astral Docs](https://docs.astral.sh/uv/concepts/projects/sync/)).

To pull the latest `dnadesign` and refresh your environment:

```bash
uv sync --upgrade-package dnadesign
```

This will update `uv.lock` (bumping the pinned commit SHA for `dnadesign`) and then re-sync `.venv` to match. Commit the updated `uv.lock` so everyone gets the same version.

---

@e-south
