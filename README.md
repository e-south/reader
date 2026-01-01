[![CI](https://github.com/e-south/reader/actions/workflows/ci.yaml/badge.svg)](https://github.com/e-south/reader/actions/workflows/ci.yaml)

## reader

**reader** is a workbench for experimental data, where the unit of work is an **experimental directory**: you put raw inputs there, keep notebooks next to them, and write (`outputs/`) in the same place. **reader** includes a plugin-based pipeline runner (see [`docs/pipeline.md`](./docs/pipeline.md) and [`docs/plugins.md`](./docs/plugins.md)), but more broadly these workspaces are where you can iterate on one experiment with a mix of:

- **lightweight utilities** (`ls`, `run`, …)
- **repeatable steps** (via `config.yaml` + CLI)
- **exploratory work** (via marimo notebooks)

### Contents

- [Repo layout](#repo-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI workbench commands](#cli-workbench-commands)
- [Docs](#docs)
- [Maintaining dependencies](#maintaining-dependencies)
- [Upgrading dependencies](#upgrading-dependencies)

---

### Repo layout

**reader** is basically two things: a place for experiments, and a library/CLI that helps you analyze them.

```bash
reader/
├─ experiments/             # experiment workbench (configs + data + results)
│  └─ 2025/
│     └─ 20250512_my_experiment/
│     ├─ config.yaml        # pipeline spec (steps) + optional reports (plots/exports)
│     ├─ inputs/            # instrument exports (preferred)
│     ├─ metadata.csv       # optional: per-sample metadata (labels, treatments)
│     ├─ notebooks/         # optional: marimo notebooks for this experiment
│     └─ outputs/           # generated: artifacts/, plots/, manifest.json, report_manifest.json, reader.log
│
└─ src/
   └─ reader/               # optional: library for running config.yaml-driven workflows across experiments
      ├─ core/              # shared commands: run/explain/validate/ls/artifacts
      ├─ parsers/           # implement instrument parsers once (raw → tidy), reuse across experiments
      ├─ plugins/           # thin adapters that expose parsers/domain/plotting to config.yaml steps
      │  ├─ ingest/         # raw → tidy artifacts (canonical tables)
      │  ├─ merge/          # tidy + metadata/table joins
      │  ├─ transform/      # reusable cleanups + derived columns/channels
      │  ├─ plot/           # optional shared plotting steps (reports)
      │  ├─ export/         # report exports (csv/tsv/excel)
      │  └─ validator/      # optional schema gates / normalizers (coercion, shape checks)
      ├─ domain/            # reusable domain helpers (math + transforms)
      ├─ plotting/          # plotting helpers used by plot plugins
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

##### 1. Initialize an experiment directory (recommended)

```bash
uv run reader init experiments/2025/20250512_my_experiment --preset plate_reader/basic
```

This creates:

```
config.yaml
inputs/
outputs/
notebooks/
```

Optionally add a metadata template:

```bash
uv run reader init experiments/2025/20250512_my_experiment --preset plate_reader/basic --metadata sample_map
```

Optionally include report presets (plots/exports):

```bash
uv run reader init experiments/2025/20250512_my_experiment \
  --preset plate_reader/basic \
  --report-preset plots/plate_reader_yfp_full
```

##### 2. Or create a folder manually

```bash
mkdir -p experiments/2025/20250512_my_experiment/{inputs,notebooks,outputs}
```

Drop experimental data into `inputs/`.

If a single experiment uses multiple instruments, you may subdivide:

```
inputs/
  plate_reader/
  cytometer/
```

##### 2. Run CLI steps (optional, repeatable)

If the experiment has a `config.yaml`:
```bash
uv run reader explain experiments/my_experiment/config.yaml
uv run reader run     experiments/my_experiment/config.yaml
uv run reader report  experiments/my_experiment/config.yaml
```

---

#### CLI workbench commands

The CLI is a set of helpers for the workspace.

```bash
reader ls --root experiments
reader init <PATH> --preset <NAME>
reader plugins
reader presets
reader contracts
reader config <CONFIG or INDEX>
reader steps <CONFIG or INDEX>
reader artifacts <CONFIG or INDEX>
reader report <CONFIG or INDEX>
```

Common workflow helpers:

- `reader ls [--root DIR]` — list experiments (finds `**/config.yaml`).
- `reader init PATH --preset NAME` — scaffold a new experiment directory.
- `reader explain CONFIG|DIR|INDEX` — show the plan (what would run).
- `reader validate CONFIG|DIR|INDEX` — validate the config + plugin configs.
- `reader config CONFIG|DIR|INDEX` — print the expanded config (presets + overrides applied).
- `reader config --schema` — print the JSON schema for config.yaml.
- `reader presets NAME --write` — emit a minimal config.yaml using a preset.
- `reader run CONFIG|DIR|INDEX [--step STEP] [--resume-from ID] [--until ID]` — execute (sliceable).
- `reader run --with-report ...` — run pipeline then reports.
- `reader report CONFIG|DIR|INDEX [--step STEP]` — run report steps only (plots/exports).
- `reader run-step STEP --config CONFIG|DIR|INDEX` — run exactly one step using existing artifacts.
- `reader artifacts CONFIG|DIR|INDEX` — list the latest artifact locations (manifest-backed).
- `reader plugins` — show discovered plugins (built-ins + entry points).
- `reader presets [NAME]` — list built-in presets or show expanded steps.
- `reader contracts [ID]` — list built-in schemas or show details for one.

**Note:** `ls` only lists experiments that have a `config.yaml`. Notebooks can exist without configs,
but then they won’t be discoverable via ls.

`reader` appends a short command log to `JOURNAL.md` inside each experiment when you run `explain`,
`validate`, `run`, or `run-step`.

---

#### Docs

Core references:

- `docs/pipeline.md` — configs, steps, and artifact flow
- `docs/plugins.md` — plugin contracts and extension points
- `docs/marimo_reference.md` — optional notebook workflow (if you use marimo)
- `docs/sfxi_vec8.md` — SFXI vec8 label (reader-side)
- `docs/logic_symmetry.md` — logic-symmetry plot usage

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

This will update `uv.lock`** (bumping the pinned commit SHA for `dnadesign`) and then re-sync `.venv` to match. Commit the updated `uv.lock` so everyone gets the same version.

---

@e-south
