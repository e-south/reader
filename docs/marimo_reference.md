# Marimo notebooks (optional)

Marimo notebooks are an optional, lightweight way to explore data alongside an experiment. This doc keeps the workflow concise and operational.

## Quickstart (project env)

```bash
uv sync --locked --group notebooks
uv run marimo edit notebook.py
```

This runs marimo inside the project environment so notebooks can import `reader` and dependencies from `uv.lock`.

## Sandboxed notebooks (portable)

```bash
uvx marimo edit --sandbox notebook.py
uv run notebook.py
```

To use your local `reader` checkout inside a sandboxed notebook:

```bash
uv add --script path/to/notebook.py . --editable
```

Add/remove notebook-specific deps:

```bash
uv add    --script notebook.py numpy
uv remove --script notebook.py numpy
```

## Minimal rules (pragmatic)

- Import `marimo as mo` in the first cell.
- Don’t re‑declare variables across cells (marimo is reactive).
- Define UI elements in one cell and read `.value` in a different cell.
- Validate inputs early (required columns, types, timepoints).
- Let the last expression be the object you want to display.

For details beyond this page, use the official Marimo docs.
