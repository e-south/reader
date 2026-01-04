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

Validate schema and wiring:

```bash
reader validate CONFIG|DIR|INDEX
```

Validate input files (checks `reads: file:` paths):

```bash
reader validate CONFIG|DIR|INDEX --files
```

Inspect the resolved plan without execution:

```bash
reader explain CONFIG|DIR|INDEX
```

---

## Pipeline (artifacts)

Run the pipeline section only (produces artifacts + `outputs/manifest.json`):

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

---

## Plots

Run plot specs only:

```bash
reader plot CONFIG|DIR|INDEX --mode save
```

`--mode` is required (unless you use `--list`) and must be `save` or `notebook`.

List resolved plot spec ids:

```bash
reader plot CONFIG|DIR|INDEX --list
```

Filter plots:

```bash
reader plot CONFIG|DIR|INDEX --mode save --only plot_ts --only plot_snapshot
reader plot CONFIG|DIR|INDEX --mode save --exclude plot_debug
```

Plot-focused notebook (no plot plugin execution):

```bash
reader plot CONFIG|DIR|INDEX --mode notebook --only plot_ts --edit
```

Add `--edit` to open the generated notebook immediately.

Ad-hoc overrides (plot/export only):

```bash
reader plot CONFIG|DIR|INDEX --mode save --only plot_ts --input df=ratio_yfp_od600/df
reader plot CONFIG|DIR|INDEX --mode save --only plot_ts --set with.time=6.0
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

Scaffold a marimo notebook (no pipeline execution):

```bash
reader notebook CONFIG|DIR|INDEX --preset notebook/basic --edit
```

Add `--edit` to open the notebook immediately (uses `uv run marimo edit`).

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

List artifacts from `outputs/manifest.json`:

```bash
reader artifacts CONFIG|DIR|INDEX
```

---

@e-south
