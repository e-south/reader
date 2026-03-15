# reader specification

This document is the developer‑oriented source of truth for how **reader** is structured, how configs map to execution, and how dependencies are managed.

---

### Scope

- **Experiment directory** = unit of work.
- **Pipeline steps** produce dataframe records; **plot/export specs** render file outputs tracked in the record catalog.
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
    core/               # config package, CLI, records, engine package
      config/           # parse-time schema + config loading/normalization
      engine/           # planning, validation, contracts, runtime execution
      notebooks/        # notebook template catalog + scaffold writer
      records/          # record catalog store + dataset discovery helpers
      workbench/        # plugin + spec ontology, semantic catalogs, spec materialization
    plugins/            # ingest/merge/transform/plot/export/validator
    io/                 # instrument parsing (raw -> tidy)
    lib/                # reusable domain logic
      microplates/      # plotting library split into support/, panels/, figure packages
        snapshot_barplot/
        snapshot_heatmap/
    tests/
```

---

### Contracts

Plugins declare input/output contracts (schema identifiers). The engine:
- asserts required inputs are present
- validates declared outputs
- fails fast on mismatches (unless runtime strictness is relaxed, in which case mismatches are logged as warnings)

Built‑in contracts live in `src/reader/core/contracts/`.
They are organized by semantic domain (`generic.py`, `plate_reader.py`,
`cytometry.py`, `analysis.py`) so lineage and responsibility stay explicit
instead of accumulating in one registry file.

The engine is now organized as a package instead of a monolithic module:

- `core/engine/planning.py` owns explain/plan rendering
- `core/engine/validation.py` owns config/reference checks
- `core/engine/contracts.py` owns runtime contract enforcement
- `core/engine/inputs.py` owns dataframe-record/file input resolution
- `core/engine/runtime.py` owns execution orchestration

That split keeps plan-time semantics, runtime semantics, and filesystem concerns orthogonal.

The plugin surface follows the same idea:

- `core/workbench/ontology.py` defines the workbench vocabulary for plugins
  (`category`, `domain`, `family`, `summary`, `tags`)
- `core/workbench/catalog.py` provides semantic indexes over installed plugins
- `core/workbench/specs.py` materializes `pipeline`, `plot`, `export`, and
  `notebook` config entries into one shared `WorkbenchSpec` model so planning,
  validation, runtime, and CLI inspection operate on the same semantic shape
- `core/registry.py` owns plugin discovery and registration, but semantic
  grouping now lives in the workbench catalog instead of being inferred only
  from `uses` strings or filesystem layout
- `core/notebooks/catalog.py` owns the notebook template registry, while
  `core/notebooks/scaffold.py` owns notebook file generation

The plotting library follows the same direction:

- `lib/microplates/support/` for selection, grouping, ordering, and file helpers
- `lib/microplates/support/emission.py` for shared figure emission semantics
- `lib/microplates/panels/` for axes-level drawing primitives
- figure-specific packages such as `snapshot_barplot/` and `snapshot_heatmap/`
  for figure planning plus render orchestration

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
