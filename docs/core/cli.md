# CLI reference

Use the CLI in progressive disclosure order:

1. `uv run reader ls` to find experiments.
2. `uv run reader init` to scaffold a new experiment from a protocol.
3. `uv run reader inspect` to see one experiment’s authoring bindings, inputs, pipeline chain, plots, exports, and current outputs.
4. `uv run reader steps` for a compact pipeline-only daisy chain.
5. `uv run reader explain` for the full compiled runtime plan.
6. `uv run reader run`, `uv run reader plot`, `uv run reader export`, and `uv run reader notebook` to materialize outputs.

`uv run reader` commands accept a config path, experiment directory, or an index from `uv run reader ls` (shown below as `CONFIG|DIR|INDEX`).

```bash
uv run reader <command> CONFIG|DIR|INDEX [options]
```

If `CONFIG|DIR|INDEX` is omitted, `uv run reader` searches upward from the current working directory for
`config.yaml`. If a numeric index is provided, it is resolved against the nearest `experiments/`
directory (or `./experiments` if none is found) using the same default experiment inventory as
`uv run reader ls`; indices shown by `uv run reader ls --all` are accepted too.

---

## Discovery

List experiments:

```bash
uv run reader ls --root experiments
```

Show protocol ids, selected-plan summaries, and current output counts:

```bash
uv run reader ls --root experiments --details
```

Add readiness state so the inventory tells you whether each experiment is
draft/template, blocked, ready to run, or already has a record catalog:

```bash
uv run reader ls --root experiments --details --readiness
```

Emit the same inventory as JSON for agents or automation:

```bash
uv run reader ls --root experiments --details --format json
uv run reader ls --root experiments --details --readiness --format json
```

The JSON payload uses explicit `catalog`, `selection`, `summary`, and
`experiments` blocks so agents do not need to reconstruct fleet state by
walking every row or guessing which filters produced the current view.
When `--readiness` is enabled, `selection.readiness` is `true`, each experiment
entry gains a `readiness` block, and `summary.by_readiness` counts the fleet by
`config_error`, `draft`, `template`, `dependency_blocked`, `blocked`, `runnable`,
`legacy_outputs_present`, or `records_ready`.

Filter the inventory down to one assay family, one lifecycle, or just broken configs:

```bash
uv run reader ls --root experiments --details --protocol plate_reader/dual_reporter_screen
uv run reader ls --root experiments --details --protocol plate_reader/single_reporter_screen
uv run reader ls --root experiments --details --protocol plate_reader/retron_sponge_screen
uv run reader ls --root experiments --details --lifecycle draft
uv run reader ls --root experiments --details --status config_error
```

Include scaffold/template directories too:

```bash
uv run reader ls --root experiments --all
```

If `--root` is omitted, `uv run reader` auto-detects the nearest `experiments/` directory.

Inspect plugins, protocols, and notebook templates:

```bash
uv run reader plugins
uv run reader plugins --category plot
uv run reader plugins --protocol plate_reader/dual_reporter_screen --category transform
uv run reader plugins --protocol plate_reader/single_reporter_screen --category plot
uv run reader plugins --protocol plate_reader/retron_sponge_screen --category transform
uv run reader protocols
uv run reader protocols plate_reader/dual_reporter_screen
uv run reader protocols plate_reader/single_reporter_screen
uv run reader protocols plate_reader/retron_sponge_screen
uv run reader protocols <protocol-id> --example-config
uv run reader protocols --family screen_analysis
uv run reader protocols --family matched_control_screen
uv run reader notebook --list-templates
```

Scaffold a new experiment from a protocol:

```bash
uv run reader init ./experiments/20260317_new_assay --protocol <protocol-id>
```

Use `plate_reader/dual_reporter_screen` for CFP/YFP-style dual-reporter panels. Use `plate_reader/single_reporter_screen` for RFP-or-other single-reporter panels normalized to a configured denominator. Use `plate_reader/retron_sponge_screen` when the assay contract depends on matched same-sensor tetO controls plus compiled burden, leakiness, induced-effect, and cross-sensor ranking nodes.

For the matched-control sponge workflow itself, use the [Retron sponge screen guide](../guides/retron_sponge_screen.md). That guide maps the direct-ratio analysis sequence, the compiled semantic tables, and the retron-specific plot/export surface.

Inspect one experiment end to end:

```bash
uv run reader inspect CONFIG|DIR|INDEX
```

Emit the experiment as layered JSON with `authoring`, `semantics`, and
`implementation`:

```bash
uv run reader inspect CONFIG|DIR|INDEX --format json
```

List just the pipeline chain and bindings:

```bash
uv run reader steps CONFIG|DIR|INDEX
uv run reader steps CONFIG|DIR|INDEX --format json
```

Guided walkthrough:

```bash
uv run reader demo
```

Protocol descriptions are the main discovery surface for user-facing outputs:

- `uv run reader protocols <id>` lists the protocol input/analysis surface, plot profiles, plot outputs, export artifacts, and the default compiled pipeline/plot/export implementations behind them.
- `uv run reader protocols <id> --example-config` prints a starter `reader/v7` YAML outline.
- `uv run reader protocols <id> --format json` exposes the assay in three explicit layers: `authoring`, `semantics`, and `implementation`.
- `uv run reader config ... --format json`, `uv run reader steps ... --format json`, `uv run reader inspect ... --format json`, and `uv run reader explain ... --format json` use shared `authoring`, `semantics`, and `implementation` layers for one bound experiment, plus command-specific envelope fields.
- `uv run reader protocols <id>`, `uv run reader config`, `uv run reader steps`, `uv run reader inspect`, and `uv run reader explain` surface a semantic program with explicit execution status for controls, windows, metrics, and ranking nodes so users can see what is compiled today versus what remains descriptive-only.
- `uv run reader plugins --protocol <id> --category transform|plot|export|ingest` scopes the registry to the plugins a protocol actually uses by default, and JSON mode adds explicit `selection` plus ontology summaries by category, domain, and family.
- `uv run reader ls --details` is the scalable workbench inventory view: protocol id, selected runtime plan summary, generated output summary, and explicit config-error state.
- `uv run reader ls --details --readiness` adds preflight-aware state so users and agents can see whether an experiment is draft/template, blocked by dependencies or files, runnable, only has legacy outputs, or already has records without separately composing `validate`, `run`, and `records`.
- `uv run reader inspect ...` shows the experiment root, bound authoring values, inputs/resources, transform chain, selected plot outputs, export artifacts, current generated outputs, and the latest record catalog.
- `uv run reader inspect ...` now also carries readiness under `implementation`, including preflight status, record-catalog presence, concrete capabilities, and suggested next commands.
- `uv run reader config ... --format json` keeps the full `reader/v7` document under `authoring`, then shows assay semantics and the fully compiled runtime chain beside it.
- `uv run reader steps ... --format json` keeps the same top-level contract but narrows `implementation` to the pipeline slice and its bindings.
- `uv run reader plot ... --list` and `uv run reader export ... --list` show the concrete outputs selected after protocol compilation, and JSON mode adds explicit `catalog`, `selection`, and output-summary blocks so agents do not need to reconstruct registry shape from raw rows.

For agent harnesses and scripted audits, the discovery commands support a shared
machine-readable contract:

```bash
uv run reader ls --root experiments --details --status config_error --format json
uv run reader protocols plate_reader/dual_reporter_screen --format json
uv run reader protocols plate_reader/single_reporter_screen --format json
uv run reader protocols plate_reader/retron_sponge_screen --format json
uv run reader plugins --protocol plate_reader/dual_reporter_screen --category transform --format json
uv run reader plugins --protocol plate_reader/single_reporter_screen --category plot --format json
uv run reader plugins --protocol plate_reader/retron_sponge_screen --category transform --format json
uv run reader records CONFIG|DIR|INDEX --format json
```

These JSON payloads now carry upstream producer and contract-surface metadata
for record bindings. `uv run reader protocols`, `uv run reader config`, `uv run reader steps`,
`uv run reader inspect`, and `uv run reader explain` all separate their machine-readable
surface into `authoring`, `semantics`, and `implementation`, and the
semantic-program block includes explicit `compiled` vs `descriptive_only`
assay-node status. `semantics.program.summary` adds compiled/descriptive
coverage counts, so a consumer can see both the runtime chain and how much of
the assay semantic surface is implemented today. `uv run reader records --format json`
is the companion result-inventory surface for one experiment: it includes the
experiment identity, manifest path, summary counts by record kind and producer,
and optional revision counts when `--all` is requested. `uv run reader plugins
--format json` keeps registry filters in `selection` and ontology totals in
`summary`, while `uv run reader plot --list --format json` and `uv run reader export --list
--format json` do the same for resolved output specs. `uv run reader ls --details
--readiness --format json` is the fleet-level preflight surface, and `uv run reader
inspect --format json` embeds the same readiness view for one experiment under
`implementation.readiness`.

---

## Configuration + validation

Inspect the experiment summary before reading the lower-level plan:

```bash
uv run reader inspect CONFIG|DIR|INDEX
```

The human view now includes a readiness panel so you can see, in one place,
whether the config is blocked by files or dependencies, already has records,
and which next command is appropriate.

Print the compiled config/IR:

```bash
uv run reader config CONFIG|DIR|INDEX
```

Print the config as JSON:

```bash
uv run reader config CONFIG|DIR|INDEX --format json
```

In JSON mode, `authoring` is the full `reader/v7` document, while
`implementation` carries the compiled plan.

Validate schema, wiring, and inputs:

```bash
uv run reader validate CONFIG|DIR|INDEX
uv run reader validate CONFIG|DIR|INDEX --format json
```

In JSON mode, `uv run reader validate` keeps the preflight mode in `selection`,
then separates overall status/counts into `summary` from file-check details in
`validation`. `uv run reader validate --no-files --format json` still reports
declared file and auto-root counts even when the checks are skipped.

If you want the same preflight signal while browsing the whole workbench, use
`uv run reader ls --details --readiness`. If you want the readiness view beside
one experiment’s compiled plan and current outputs, use `uv run reader inspect`.

Skip file checks (config-only):

```bash
uv run reader validate CONFIG|DIR|INDEX --no-files
uv run reader validate CONFIG|DIR|INDEX --no-files --format json
```

Inspect the resolved plan without execution:

```bash
uv run reader explain CONFIG|DIR|INDEX
uv run reader explain CONFIG|DIR|INDEX --format json
```

---

## Pipeline (records)

Run the pipeline section only (produces dataframe records + `outputs/manifests/records.json`):

```bash
uv run reader run CONFIG|DIR|INDEX
uv run reader run CONFIG|DIR|INDEX --dry-run --format json
```

Slice the pipeline:

```bash
uv run reader run CONFIG|DIR|INDEX --from step_a --until step_c
uv run reader run CONFIG|DIR|INDEX --only step_b
uv run reader run CONFIG|DIR|INDEX --from step_a --until step_c --dry-run --format json
```

`uv run reader run` fails fast if `--from` comes after `--until` in pipeline order.

Inspect the emitted record catalog:

```bash
uv run reader records CONFIG|DIR|INDEX
uv run reader records CONFIG|DIR|INDEX --format json
uv run reader records CONFIG|DIR|INDEX --all --format json
```

In JSON mode, `uv run reader records` keeps experiment identity at the top level, then
adds the record-manifest path, a summary by record kind and producer, and the
latest record entries. `--all` does not dump every historical revision; it adds
per-record revision counts and a total revision summary so the surface stays
compact.

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
uv run reader plot CONFIG|DIR|INDEX
```

Run plots for all experiments in a year (expects `experiments/YYYY`):

```bash
uv run reader plot --year 2025
```

Override the experiments root when using `--year`:

```bash
uv run reader plot --year 2025 --root /path/to/experiments
```

List resolved semantic plot outputs and their upstream dataframe bindings:

```bash
uv run reader plot CONFIG|DIR|INDEX --list
uv run reader plot CONFIG|DIR|INDEX --list --format json
```

In JSON mode, `uv run reader plot --list` keeps the bound experiment at the top level,
then adds `catalog`, `selection`, and `summary` blocks before the resolved
`plots` entries.

Dry-run a plot plan without executing:

```bash
uv run reader plot CONFIG|DIR|INDEX --dry-run
```

Filter plots:

```bash
uv run reader plot CONFIG|DIR|INDEX --only raw_kinetics --only ratio_heatmap
uv run reader plot CONFIG|DIR|INDEX --exclude value_distributions
```

Ad-hoc overrides (plot/export only):

```bash
uv run reader plot CONFIG|DIR|INDEX --only raw_kinetics --input 'df={record: ratio_yfp_od600/df}'
uv run reader plot CONFIG|DIR|INDEX --only endpoint_by_condition --set with.time=6.0
```

`--input` expects a structured YAML/JSON binding such as `{record: ...}`,
`{file: ...}`, or `{resource: ...}`.
`--set` paths must start with `reads.`, `with.`, or `writes.`.

---

## Exports

Run export specs only:

```bash
uv run reader export CONFIG|DIR|INDEX
```

List resolved semantic export artifacts and their upstream dataframe bindings:

```bash
uv run reader export CONFIG|DIR|INDEX --list
uv run reader export CONFIG|DIR|INDEX --list --format json
```

In JSON mode, `uv run reader export --list` mirrors the same shape as plot listings:
experiment identity, then `catalog`, `selection`, `summary`, and resolved
`exports` entries.

Dry-run an export plan without executing:

```bash
uv run reader export CONFIG|DIR|INDEX --dry-run
```

Filter exports:

```bash
uv run reader export CONFIG|DIR|INDEX --only crosstalk_pairs_table
uv run reader export CONFIG|DIR|INDEX --exclude logic_summary_workbook
```

Ad-hoc overrides:

```bash
uv run reader export CONFIG|DIR|INDEX --only crosstalk_pairs_table --set with.path="exports/crosstalk_pairs.csv"
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
uv run reader notebook CONFIG|DIR|INDEX
```

Choose a template explicitly:

```bash
uv run reader notebook CONFIG|DIR|INDEX --template notebook/eda
```

Allow explicit record scanning when the canonical catalog is missing:

```bash
uv run reader notebook CONFIG|DIR|INDEX --scan-records
```

Name the notebook explicitly:

```bash
uv run reader notebook CONFIG|DIR|INDEX --name EDA_custom.py
```

Launch modes:

- `--mode edit` (default): open Marimo editor
- `--mode run`: run as a read-only app
- `--mode none`: create only (no launch)

See templates:

```bash
uv run reader notebook --list-templates
```

Overwrite an existing notebook:

```bash
uv run reader notebook CONFIG|DIR|INDEX --template notebook/basic --force
```

Create a new notebook with a numeric suffix if the name already exists:

```bash
uv run reader notebook CONFIG|DIR|INDEX --new
```

Regenerate a notebook in-place:

```bash
uv run reader notebook CONFIG|DIR|INDEX --refresh
```

Filter plots injected into a template that declares plot-filter capability
(currently `notebook/eda`):

```bash
uv run reader notebook CONFIG|DIR|INDEX --template notebook/eda --only raw_kinetics
uv run reader notebook CONFIG|DIR|INDEX --template notebook/eda --exclude value_distributions
```

---

## Introspection

List pipeline steps (resolved):

```bash
uv run reader steps CONFIG|DIR|INDEX
```

List workbench records from `outputs/manifests/records.json`:

```bash
uv run reader records CONFIG|DIR|INDEX
```

Show record history counts:

```bash
uv run reader records CONFIG|DIR|INDEX --all
```

List step ids and plugins:

```bash
uv run reader steps CONFIG|DIR|INDEX
uv run reader plugins
```

---

@e-south
