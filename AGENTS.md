## `reader` for agents

### What this repo is
`reader` is a Python package + CLI for working with experimental data. The repo is both:
1) A **workbench** of experiment directories under `experiments/`, and
2) A reusable **library/CLI** under `src/reader/` that runs config-driven pipelines across experiments.

The unit of work is an **experiment directory**:
- put raw inputs in `inputs/`
- keep hand-authored notebooks in `notebooks/`; `reader notebook` scaffolds land in the experiment's `outputs/notebooks/`
- write generated results to the experiment's `outputs/`

### Scope & safety
- Work only inside this repository unless I explicitly add directories.
- Prefer small, reviewable changes. Avoid large refactors unless asked.
- Before edits: summarize what you plan to change and why.
- After edits: show a diff summary and list commands run.

### Repo layout
- Source code: `src/reader/`
- Tests: `src/reader/tests/`
- Docs: `docs/`
- Skills: `skills/` (repo-specific agent workflows)
- Experiments: `experiments/` (experiment directories)

### Agent map
- Keep `AGENTS.md` short. Detailed workflow guidance belongs in `docs/`.
- For Data Operations Plan classification, registry/docs alignment, or DOP maintenance, use `docs/guides/data_operations_plan.md` and `skills/reader-data-operations-plan/SKILL.md`.
- For new experiment intake, metadata mapping, or Google Drive-backed workspace creation, use `docs/guides/experiment_bootstrap.md` and `skills/reader-experiment-bootstrap/SKILL.md`.
- Treat `docs/guides/experiment_bootstrap.md` as the primary workflow and the repo-local skill as the router to it.
- For workbench architecture, information-architecture gardening, or maintainer-facing surface hardening, use `docs/guides/workbench_gardening.md` and `skills/reader-workbench-gardening/SKILL.md`.
- Repo `pytest` markers cover tracked fixtures. To audit local experiments, run `uv run reader audit experiments [--years <yyyy>]`.

#### Generated vs hand-edited content
- Treat `experiments/**/outputs/` as **generated**.
- **Do not hand-edit generated artifacts** in experiment `outputs/` directories (plots, manifests, logs, etc.).
- Do not create a repository-root `outputs/` directory. A cross-experiment aggregate is itself an experiment and publishes through that experiment's configured `outputs/` directory.
  - If output content is wrong, fix the pipeline code/config and re-run to regenerate.
- Keep disposable repository-local scratch under `.tmp/`; `tmp/` is not a valid top-level namespace.
- Ask before committing any data or generated outputs.

### Environment & tooling (uv)
This project uses `uv`:
- `uv sync` installs dependencies from the lockfile into `.venv/`
- `uv run <cmd>` runs commands inside the project environment.

#### Setup (dev + notebooks)
```bash
uv sync --locked --group dev --group notebooks
```

## Commands

### CLI help

```bash
uv run reader --help
uv run reader ls --help
```

#### Tests, lint, format

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format . --check
```

### Notebooks (marimo)

Hand-authored notebooks should live in `experiments/<exp>/notebooks/`.
`reader notebook` writes generated scaffolds under `outputs/notebooks/`.

There is additional, notebook-specific guidance in:

* `docs/guides/marimo_reference.md` (longer reference: UI widgets + examples)

### Definition of Done (for changes in this repo)

* [ ] Diff reviewed (`/diff` in Codex or VS Code Source Control)
* [ ] Tests pass: `uv run pytest -q` (or explain why not applicable)
* [ ] Docs integrity checked when docs or routing changed: `uv run reader maintain docs`
* [ ] Lint/format pass: `uv run ruff check .` and `uv run ruff format . --check`
* [ ] `git diff --check` passes
* [ ] No hand edits to generated outputs (`experiments/**/outputs/`)
* [ ] Commit message explains *why* (not just what)
* [ ] If pushing to GitHub: changes live on a branch and are ready for a Draft PR
