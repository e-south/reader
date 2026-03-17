# CLI reference

Use the CLI in progressive disclosure order:

1. `reader ls` to find experiments.
2. `reader init` to scaffold a new experiment from a protocol.
3. `reader inspect` to see one experiment’s authoring bindings, inputs, pipeline chain, plots, exports, and current outputs.
4. `reader steps` for a compact pipeline-only daisy chain.
5. `reader explain` for the full compiled runtime plan.
6. `reader run`, `reader plot`, `reader export`, and `reader notebook` to materialize outputs.

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

Show protocol ids, selected-plan summaries, and current output counts:

```bash
reader ls --root experiments --details
```

Emit the same inventory as JSON for agents or automation:

```bash
reader ls --root experiments --details --format json
```

The JSON payload includes a top-level `summary` block with counts by protocol,
status, and output presence so agents do not need to reconstruct fleet state by
walking every row.

Filter the inventory down to one assay family or just broken configs:

```bash
reader ls --root experiments --details --protocol plate_reader/dual_reporter_screen
reader ls --root experiments --details --status config_error
```

Include scaffold/template directories too:

```bash
reader ls --root experiments --all
```

If `--root` is omitted, `reader` auto-detects the nearest `experiments/` directory.

Inspect plugins, protocols, and notebook templates:

```bash
reader plugins
reader plugins --category plot
reader plugins --protocol plate_reader/dual_reporter_screen --category transform
reader protocols
reader protocols plate_reader/dual_reporter_screen
reader protocols plate_reader/dual_reporter_screen --example-config
reader protocols --family screen_analysis
reader notebook --list-templates
```

Scaffold a new experiment from a protocol:

```bash
reader init ./experiments/20260317_new_assay --protocol plate_reader/dual_reporter_screen
```

Inspect one experiment end to end:

```bash
reader inspect CONFIG|DIR|INDEX
```

Emit the bound authoring surface, generated outputs, record catalog, and
compiled chain as JSON:

```bash
reader inspect CONFIG|DIR|INDEX --format json
```

List just the pipeline chain and bindings:

```bash
reader steps CONFIG|DIR|INDEX
reader steps CONFIG|DIR|INDEX --format json
```

Guided walkthrough:

```bash
reader demo
```

Protocol descriptions are the main discovery surface for user-facing outputs:

- `reader protocols <id>` lists the protocol input/analysis surface, plot profiles, plot outputs, export artifacts, and the default compiled pipeline/plot/export implementations behind them.
- `reader protocols <id> --example-config` prints a starter `reader/v7` YAML outline.
- `reader protocols <id> --format json` exposes the same semantic surface plus the compiled default runtime chain in machine-readable form.
- `reader protocols <id>`, `reader inspect`, and `reader explain` now surface a protocol semantic program, with explicit execution status for controls, windows, metrics, and ranking nodes so users can see what is compiled today versus what remains descriptive-only.
- `reader plugins --protocol <id> --category transform|plot|export|ingest` scopes the registry to the plugins a protocol actually uses by default.
- `reader ls --details` is the scalable workbench inventory view: protocol id, selected runtime plan summary, generated output summary, and explicit config-error state.
- `reader inspect ...` shows the experiment root, bound authoring values, inputs/resources, transform chain, selected plot outputs, export artifacts, current generated outputs, and the latest record catalog.
- `reader plot ... --list` and `reader export ... --list` show the concrete outputs selected after protocol compilation.

For agent harnesses and scripted audits, the discovery commands support a shared
machine-readable contract:

```bash
reader ls --root experiments --details --status config_error --format json
reader protocols plate_reader/dual_reporter_screen --format json
reader plugins --protocol plate_reader/dual_reporter_screen --category transform --format json
reader records CONFIG|DIR|INDEX --format json
```

These JSON payloads now carry upstream producer and contract-surface metadata
for record bindings, and the protocol/inspect/explain routes include a semantic
program block with explicit `compiled` vs `descriptive_only` assay-node status,
so a consumer can see both the runtime chain and the assay semantics it does or
does not implement today.

---

## Configuration + validation

Inspect the experiment summary before reading the lower-level plan:

```bash
reader inspect CONFIG|DIR|INDEX
```

Print the compiled config/IR:

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
reader validate CONFIG|DIR|INDEX --format json
```

Skip file checks (config-only):

```bash
reader validate CONFIG|DIR|INDEX --no-files
reader validate CONFIG|DIR|INDEX --no-files --format json
```

Inspect the resolved plan without execution:

```bash
reader explain CONFIG|DIR|INDEX
reader explain CONFIG|DIR|INDEX --format json
```

---

## Pipeline (records)

Run the pipeline section only (produces dataframe records + `outputs/manifests/records.json`):

```bash
reader run CONFIG|DIR|INDEX
reader run CONFIG|DIR|INDEX --dry-run --format json
```

Slice the pipeline:

```bash
reader run CONFIG|DIR|INDEX --from step_a --until step_c
reader run CONFIG|DIR|INDEX --only step_b
reader run CONFIG|DIR|INDEX --from step_a --until step_c --dry-run --format json
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

List resolved semantic plot outputs and their upstream dataframe bindings:

```bash
reader plot CONFIG|DIR|INDEX --list
reader plot CONFIG|DIR|INDEX --list --format json
```

Dry-run a plot plan without executing:

```bash
reader plot CONFIG|DIR|INDEX --dry-run
```

Filter plots:

```bash
reader plot CONFIG|DIR|INDEX --only raw_kinetics --only ratio_heatmap
reader plot CONFIG|DIR|INDEX --exclude value_distributions
```

Ad-hoc overrides (plot/export only):

```bash
reader plot CONFIG|DIR|INDEX --only raw_kinetics --input 'df={record: ratio_yfp_od600/df}'
reader plot CONFIG|DIR|INDEX --only endpoint_by_condition --set with.time=6.0
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

List resolved semantic export artifacts and their upstream dataframe bindings:

```bash
reader export CONFIG|DIR|INDEX --list
reader export CONFIG|DIR|INDEX --list --format json
```

Dry-run an export plan without executing:

```bash
reader export CONFIG|DIR|INDEX --dry-run
```

Filter exports:

```bash
reader export CONFIG|DIR|INDEX --only crosstalk_pairs_table
reader export CONFIG|DIR|INDEX --exclude logic_summary_workbook
```

Ad-hoc overrides:

```bash
reader export CONFIG|DIR|INDEX --only crosstalk_pairs_table --set with.path="exports/crosstalk_pairs.csv"
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
reader notebook CONFIG|DIR|INDEX --template notebook/eda --only raw_kinetics
reader notebook CONFIG|DIR|INDEX --template notebook/eda --exclude value_distributions
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
