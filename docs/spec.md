## reader spec (architecture + developer workflow)

This document is the developer‑oriented source of truth for how **reader** is structured, how configs map to execution, and how dependencies are managed.

Quick links:
- [README](../README.md)
- [Pipeline config + deliverables](./pipeline.md)
- [CLI reference](./cli.md)
- [Plugin development](./plugins.md)

---

### Scope

- **Experiment directory** = unit of work.
- **Pipeline steps** produce artifacts; **deliverables** render plots/exports.
- **Notebooks** are optional and read outputs for interactive exploration.

---

### Repo layout

```text
reader/
  experiments/          # workbench directories (inputs, notebooks, outputs)
  docs/                 # narrative + reference docs
  src/reader/            # library + CLI
    core/               # engine, config, manifests, CLI
    plugins/            # ingest/merge/transform/plot/export/validator
    io/                 # instrument parsing (raw -> tidy)
    lib/                # reusable domain logic
    tests/
```

---

### Config + outputs

This spec defers to [pipeline.md](./pipeline.md) for config keys, execution semantics, and outputs/manifests.

---

### Contracts

Plugins declare input/output contracts (schema identifiers). The engine:
- asserts required inputs are present
- validates declared outputs
- fails fast on mismatches (unless runtime strictness is relaxed)

Built‑in contracts live in `src/reader/core/contracts.py`.

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
