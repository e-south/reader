# reader specification

This document is the developer‑oriented source of truth for how **reader** is structured, how configs map to execution, and how dependencies are managed.

---

### Scope

- **Experiment directory** = unit of work.
- **Pipeline steps** produce artifacts; **plot/export specs** render outputs (file or inline notebook).
- **Notebooks** are optional and read outputs for interactive exploration.

---

### Repo layout

```text
reader/
  experiments/          # workbench directories (inputs, notebooks, outputs)
  docs/                 # documentation (index + grouped references)
    index.md            # docs map
    core/               # core reference (CLI, pipeline, plugins, spec)
    guides/             # how-to + walkthroughs
    lib/                # library-level references
    audits/             # audits and investigations
  src/reader/            # library + CLI
    core/               # engine, config, manifests, CLI
    plugins/            # ingest/merge/transform/plot/export/validator
    io/                 # instrument parsing (raw -> tidy)
    lib/                # reusable domain logic
    tests/
```

---

### Contracts

Plugins declare input/output contracts (schema identifiers). The engine:
- asserts required inputs are present
- validates declared outputs
- fails fast on mismatches (unless runtime strictness is relaxed, in which case mismatches are logged as warnings)

Built‑in contracts live in `src/reader/core/contracts.py`.

---

### Matplotlib cache

Plotting plugins require a writable Matplotlib cache directory. `reader` sets
`MPLCONFIGDIR` automatically when plotting is needed.

Defaults:
- Commands that resolve a config/experiment (run/explain/validate/plot/export) use
  `<paths.outputs>/.cache/matplotlib`.
- Other commands that load plot plugins without a config (e.g., `reader plugins`)
  use `$XDG_CACHE_HOME/reader/matplotlib` (or `~/.cache/reader/matplotlib`).

Override with `MPLCONFIGDIR` or `READER_MPLCONFIGDIR` if you need a custom path.

---

### Dependency management (uv)

This repo uses **uv**:

```bash
uv sync --locked
```

Developer tooling (lint + tests + notebooks):

```bash
uv sync --locked --group dev --group notebooks
uv run ruff check .
uv run pytest -q
```

Add/remove dependencies:

```bash
uv add <package>
uv add --group dev <package>
uv remove <package>
```

If you edit `pyproject.toml` manually, regenerate the lockfile:

```bash
uv lock
```

---

### Upgrading dependencies

To upgrade a pinned package:

```bash
uv sync --upgrade-package <name>
```

Commit `pyproject.toml` and `uv.lock` together.

---

@e-south
