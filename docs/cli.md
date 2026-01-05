# CLI reference

`reader` commands accept a config path, experiment directory, or an index from `reader ls` (shown below as `CONFIG|DIR|INDEX`).

```bash
reader <command> CONFIG|DIR|INDEX [options]
```

---

## Discovery

List experiments (searches for `config.yaml`):

```bash
reader ls --root experiments
```

Inspect plugins and presets:

```bash
reader plugins
reader plugins --category plot
reader presets
reader presets --category plot
```

---

## Configuration + validation

Print the expanded config (presets + overrides applied):

```bash
reader config CONFIG|DIR|INDEX
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

## Pipeline (artifacts)

Run the pipeline section only (produces artifacts + `outputs/manifests/manifest.json`):

```bash
reader run CONFIG|DIR|INDEX
```

Slice the pipeline:

```bash
reader run CONFIG|DIR|INDEX --from step_a --until step_c
reader run CONFIG|DIR|INDEX --only step_b
```

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

List resolved plot spec ids:

```bash
reader plot CONFIG|DIR|INDEX --list
```

Filter plots:

```bash
reader plot CONFIG|DIR|INDEX --only plot_ts --only plot_snapshot
reader plot CONFIG|DIR|INDEX --exclude plot_debug
```

Ad-hoc overrides (plot/export only):

```bash
reader plot CONFIG|DIR|INDEX --only plot_ts --input df=ratio_yfp_od600/df
reader plot CONFIG|DIR|INDEX --only plot_ts --set with.time=6.0
```

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

Scaffold a marimo notebook (no pipeline execution). If `--preset` is omitted, the CLI
uses `notebook.preset` from config, otherwise auto-picks `notebook/plots` when plots
exist or `notebook/basic` (both presets currently scaffold the same minimal notebook):

Notebooks are written under `outputs/notebooks/`.

```bash
reader notebook CONFIG|DIR|INDEX
```

Launch modes:

- `--mode edit` (default): open Marimo editor
- `--mode run`: run as a read-only app
- `--mode none`: create only (no launch)

See presets:

```bash
reader notebook --list-presets
```

Overwrite an existing notebook:

```bash
reader notebook CONFIG|DIR|INDEX --preset notebook/basic --force
```

---

## Introspection

List pipeline steps (resolved):

```bash
reader steps CONFIG|DIR|INDEX
```

List artifacts from `outputs/manifests/manifest.json`:

```bash
reader artifacts CONFIG|DIR|INDEX
```

---

@e-south
