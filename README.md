# reader

`reader` is a workbench for analyzing experimental data and a library/CLI that provides supporting commands.


### Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [More documentation](#more-documentation)

---

### Overview

1. Place raw data and metadata in an experiment directory.

```bash
experiments/
    <exp>/
        config.yaml
        inputs/
        outputs/
```
2. Run config-driven pipelines that process data and produce structured outputs. 

```bash
outputs/
  reader.log
  manifests/
    manifest.json
    plots_manifest.json
    exports_manifest.json
  artifacts/
    <step_id>.<plugin_key>/        # first revision
      <output>.parquet
      meta.json
  plots/                           # optional; only if plot specs write figures
  exports/                         # optional; only if export specs write files
  notebooks/
```

3. Optionally generate plots and exports from those outputs.
4. Use notebooks for interactive exploration against the same outputs.

---

### Installation

This repo is managed with [**uv**](https://docs.astral.sh/uv/):

1. `pyproject.toml` declares dependencies (runtime + optional extras).
2. `uv.lock` is the fully pinned dependency graph.
3. `.venv/` is the project virtual environment.

**Two key commands:**

1. `uv sync` installs everything from the lockfile into .venv.
2. `uv run <cmd>` runs commands inside the project environment without requiring `source .venv/bin/activate`.

Install uv. (Below is for macOS/Linux, for other OSs see [here](https://docs.astral.sh/uv/getting-started/installation/).)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# ensure your uv bin dir is on PATH
```

Clone repo

```bash
git clone https://github.com/e-south/reader.git
cd reader
```

Create/sync the environment from the committed lockfile:

```bash
uv sync --locked
```

Sanity checks:

```bash
uv run python -c "import reader, pandas, pyarrow; print('ok')"
```

Dev tooling is opt-in via a dependency group.

```bash
uv sync --locked --group dev --group notebooks --group cytometry
uv run ruff --version
uv run ruff check .
uv run pytest -q
```

Optional: design rendering (Baserender via dnadesign).

```bash
uv sync --locked --extra design
```

**This project defines console scripts, which you can run via:**

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

### Quickstart

Create an experiment directory:

```bash
# From the reader root
mkdir -p experiments/<year>/my_experiment/{inputs,outputs}
```

Design and run a pipeline via `config.yaml`:

```bash
uv run reader explain experiments/my_experiment/config.yaml
uv run reader run     experiments/my_experiment/config.yaml   # pipeline (artifacts)
```

Generate plots/exports or scaffold a notebook:

```bash
uv run reader plot         experiments/my_experiment/config.yaml --list
uv run reader export       experiments/my_experiment/config.yaml --list
uv run reader notebook     experiments/my_experiment/config.yaml
```

---

### More documentation

1. [CLI reference](./docs/cli.md)
2. [Configuring pipelines](./docs/pipeline.md)
3. [Notebooks](./docs/notebooks.md)
4. [Plugin development](./docs/plugins.md)
5. [Spec / architecture](./docs/spec.md)
6. [End-to-end demo](./docs/demo.md)

---

@e-south
