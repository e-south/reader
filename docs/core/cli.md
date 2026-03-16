# CLI reference

`reader` commands accept a config path, experiment directory, or an index from `reader ls` (shown below as `CONFIG|DIR|INDEX`).

```bash
reader <command> CONFIG|DIR|INDEX [options]
```

If `CONFIG|DIR|INDEX` is omitted, `reader` searches upward from the current working directory for
`config.yaml`. If a numeric index is provided, it is resolved against the nearest `experiments/`
directory (or `./experiments` if none is found) using the same runnable-experiment listing as
`reader ls`.

---

## Discovery

List runnable experiments:

```bash
reader ls --root experiments
```

Include scaffold/template directories too:

```bash
reader ls --root experiments --all
```

If `--root` is omitted, `reader` auto-detects the nearest `experiments/` directory.

Inspect plugins, recipes, and notebook templates:

```bash
reader plugins
reader plugins --category plot
reader recipes
reader recipes --family plot_set
reader notebook --list-templates
```

Guided walkthrough:

```bash
reader demo
```

---

## Configuration + validation

Print the expanded config (recipes + overrides applied):

```bash
reader config CONFIG|DIR|INDEX
```

Print the config as JSON:

```bash
reader config CONFIG|DIR|INDEX --format json
```

Validate schema, wiring, and inputs:

```bash
reader validate CONFIG|DIR|INDEX
```

Skip file checks (config-only):

```bash
reader validate CONFIG|DIR|INDEX --no-files
```

Inspect the resolved plan without execution:

```bash
reader explain CONFIG|DIR|INDEX
```

---

## Pipeline (records)

Run the pipeline section only (produces dataframe records + `outputs/manifests/records.json`):

```bash
reader run CONFIG|DIR|INDEX
```

Slice the pipeline:

```bash
reader run CONFIG|DIR|INDEX --from step_a --until step_c
reader run CONFIG|DIR|INDEX --only step_b
```

`reader run` fails fast if `--from` comes after `--until` in pipeline order.

Useful flags:

- `--from <step_id>` / `--until <step_id>` (pipeline only)
- `--only <step_id>` (single pipeline step)
- `--dry-run`
- `--log-level <level>`
- `--compact` (use the compact progress view instead of per-step logs)

---

## Plots

Run plot specs only (saves files to `outputs/plots`):

```bash
reader plot CONFIG|DIR|INDEX
```

Run plots for all experiments in a year (expects `experiments/YYYY`):

```bash
reader plot --year 2025
```

Override the experiments root when using `--year`:

```bash
reader plot --year 2025 --root /path/to/experiments
```

List resolved plot spec ids:

```bash
reader plot CONFIG|DIR|INDEX --list
```

Dry-run a plot plan without executing:

```bash
reader plot CONFIG|DIR|INDEX --dry-run
```

Filter plots:

```bash
reader plot CONFIG|DIR|INDEX --only plot_ts --only plot_snapshot
reader plot CONFIG|DIR|INDEX --exclude plot_debug
```

Ad-hoc overrides (plot/export only):

```bash
reader plot CONFIG|DIR|INDEX --only plot_ts --input 'df={record: ratio_yfp_od600/df}'
reader plot CONFIG|DIR|INDEX --only plot_ts --set with.time=6.0
```

`--input` expects a structured YAML/JSON binding such as `{record: ...}`,
`{file: ...}`, or `{resource: ...}`.
`--set` paths must start with `reads.`, `with.`, or `writes.`.

---

## Exports

Run export specs only:

```bash
reader export CONFIG|DIR|INDEX
```

List resolved export spec ids:

```bash
reader export CONFIG|DIR|INDEX --list
```

Dry-run an export plan without executing:

```bash
reader export CONFIG|DIR|INDEX --dry-run
```

Filter exports:

```bash
reader export CONFIG|DIR|INDEX --only export_ratios
reader export CONFIG|DIR|INDEX --exclude export_debug
```

Ad-hoc overrides:

```bash
reader export CONFIG|DIR|INDEX --only export_ratios --set with.path="exports/ratios.csv"
```

---

## Notebooks

Scaffold a marimo notebook (no pipeline execution). If `--template` is omitted, the CLI
uses the first configured `notebooks.specs` entry, otherwise auto-picks a default
template from declared template capabilities:

- plot-capable template when plots exist
- cytometry EDA template when the pipeline is cytometry-shaped
- fallback basic template otherwise

Notebooks are written under `outputs/notebooks/`.

```bash
reader notebook CONFIG|DIR|INDEX
```

Choose a template explicitly:

```bash
reader notebook CONFIG|DIR|INDEX --template notebook/eda
```

Allow explicit record scanning when the canonical catalog is missing:

```bash
reader notebook CONFIG|DIR|INDEX --scan-records
```

Name the notebook explicitly:

```bash
reader notebook CONFIG|DIR|INDEX --name EDA_custom.py
```

Launch modes:

- `--mode edit` (default): open Marimo editor
- `--mode run`: run as a read-only app
- `--mode none`: create only (no launch)

See templates:

```bash
reader notebook --list-templates
```

Overwrite an existing notebook:

```bash
reader notebook CONFIG|DIR|INDEX --template notebook/basic --force
```

Create a new notebook with a numeric suffix if the name already exists:

```bash
reader notebook CONFIG|DIR|INDEX --new
```

Regenerate a notebook in-place:

```bash
reader notebook CONFIG|DIR|INDEX --refresh
```

Filter plots injected into a template that declares plot-filter capability
(currently `notebook/eda`):

```bash
reader notebook CONFIG|DIR|INDEX --template notebook/eda --only plot_ts
reader notebook CONFIG|DIR|INDEX --template notebook/eda --exclude plot_debug
```

---

## Introspection

List pipeline steps (resolved):

```bash
reader steps CONFIG|DIR|INDEX
```

List workbench records from `outputs/manifests/records.json`:

```bash
reader records CONFIG|DIR|INDEX
```

Show record history counts:

```bash
reader records CONFIG|DIR|INDEX --all
```

List step ids and plugins:

```bash
reader steps CONFIG|DIR|INDEX
reader plugins
```

---

@e-south
