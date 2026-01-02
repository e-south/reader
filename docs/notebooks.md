## Notebooks (marimo)

Use notebooks for interactive analysis and iteration. Pipelines produce artifacts; notebooks read those artifacts and explore them.

Quick links:
- [CLI reference](./cli.md)
- [Pipeline config + deliverables](./pipeline.md)
- [Marimo notebook reference](./marimo_reference.md)

---

### Scaffold a notebook (recommended)

```bash
reader explore experiments/my_experiment/config.yaml --preset eda/basic
```

Then open the notebook using the steps below.

Notes:
- `reader explore` only scaffolds the notebook; it does not run the pipeline.
- Presets include `eda/basic`, `eda/microplate` (adds time‑series + snapshot plots), and `eda/cytometry` (population distributions).

List presets:

```bash
reader explore --list-presets
```

---

### Dependencies and running

Run marimo inside the project environment:

```bash
uv sync --locked --group notebooks
uv run marimo edit experiments/my_experiment/notebooks/eda.py
```

If your experiment uses flow cytometry plugins, include the extra group:

```bash
uv sync --locked --group notebooks --group cytometry
```

---

### Output expectations

Scaffolded notebooks:
- look up artifacts via `outputs/manifest.json`
- list deliverables from `outputs/deliverables_manifest.json`
- use DuckDB to query parquet artifacts efficiently

For notebook editing rules and marimo specifics, see [docs/marimo_reference.md](./marimo_reference.md).

---

@e-south
