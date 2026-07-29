---
doc_id: reader-cli-reference
surface: cli-reference
owner: reader-maintainers
last_verified: 2026-07-28
summary: Full reader CLI command reference with discovery, execution, outputs, notebooks, and aggregate review commands.
---

# CLI reference

This page is the full CLI reference. For setup and the shortest common paths,
start with [Getting started](../guides/getting_started.md) and
[Common tasks](../guides/common_routes.md). For the operating loop and
machine-readable output, use [Preflight, run, verify](../guides/preflight_run_verify.md)
and [Automation and JSON](../guides/automation.md).

A typical order is:

1. `uv run reader ls` to find experiments.
2. `uv run reader protocols` or `uv run reader init` to choose a protocol and scaffold a new experiment.
3. `uv run reader inspect` to see one experiment’s config, inputs, pipeline chain, plots, exports, and current outputs.
4. `uv run reader steps` or `uv run reader explain` to inspect the compiled plan.
5. `uv run reader validate` to run preflight checks.
6. `uv run reader run`, `uv run reader plot`, `uv run reader export`, and `uv run reader notebook` to materialize outputs.

Manifest-backed multi-experiment response summaries use a separate explicit
lifecycle:

```bash
uv run reader init OUTPUT_EXPERIMENT --protocol workbench/generic
uv run reader response-window preflight REQUEST.yaml --format json
uv run reader response-window build REQUEST.yaml --output-experiment OUTPUT_EXPERIMENT --format json
uv run reader response-window verify OUTPUT_EXPERIMENT/outputs --format json
uv run reader response-window review OUTPUT_EXPERIMENT/outputs --mode run
uv run reader response-window promoter-evidence BUNDLE BINDINGS \
  --output-experiment EVIDENCE_EXPERIMENT --experiment-id EXPERIMENT --design-id DESIGN \
  --reduction-id REDUCTION --format json
uv run reader response-window promoter-evidence-verify EVIDENCE_EXPERIMENT/outputs --format json
```

These commands consume published experiment records. They do not bypass the
normal experiment pipeline or infer treatment semantics from file names.
The response service accepts `reader.response_window.request.v3`, publishes
`reader.response_window.bundle.v5`, and publishes promoter evidence under
`reader.response_window.promoter_evidence_bundle.v5`.
Promoter evidence additionally consumes an explicit, study-owned candidate
binding; Reader does not open the study candidate table or calculate an OPAL
objective. Its optional v2 objective overlay accepts screen-only raw components
plus a study-supplied compact objective display label and rejects production
claims.

`uv run reader` commands accept a config path, experiment directory, or an index from `uv run reader ls` (shown below as `CONFIG|DIR|INDEX`).

```bash
uv run reader <command> CONFIG|DIR|INDEX [options]
```

If `CONFIG|DIR|INDEX` is omitted, `uv run reader` searches upward from the current working directory for
`config.yaml`. If a numeric index is provided, it is resolved against the nearest `experiments/`
directory (or `./experiments` if none is found) using the same default experiment list as
`uv run reader ls`. Hidden scaffold/template entries shown by `uv run reader ls --all` must be
addressed by explicit path.

---

## Discovery

List experiments:

```bash
uv run reader ls --root experiments
```

Show protocol ids, selected step summaries, and current output counts:

```bash
uv run reader ls --root experiments --details
```

Add readiness state so the list tells you whether each experiment is
draft/template, blocked, ready to run, or has a usable non-empty records
catalog:

```bash
uv run reader ls --root experiments --details --readiness
```

Emit the same list as JSON for agents or automation:

```bash
uv run reader ls --root experiments --details --format json
uv run reader ls --root experiments --details --readiness --format json
```

The JSON payload uses explicit `catalog`, `selection`, `summary`, and
`experiments` blocks so agents do not need to reconstruct the experiment list by
walking every row or guessing which filters produced the current view.
JSON collections return 25 entries per page by default. Use `--limit N` and
replay `meta.continuation` with `--continuation TOKEN`; the summary continues
to describe the complete filtered collection. Limits may range from 1 through
100. Paging does not change the table view.
When `--readiness` is enabled, `selection.readiness` is `true`, each experiment
entry gains a `readiness` block, and `summary.by_readiness` counts experiments by
`config_error`, `draft`, `template`, `dependency_blocked`, `blocked`, `runnable`,
`uncataloged_outputs_present`, `catalog_ready`, or `records_ready`.

Within each readiness block, `records.catalog` reports whether `records.json`
exists and `records.available` reports whether it contains usable current
records. `records.verification` reports `ok`, `unverifiable`, `failed`, or
`null` when no current records exist. A valid schema-v5 catalog whose config or
build identity differs from the current environment is `catalog_ready`;
verified current evidence is `records_ready`. An empty catalog is runnable,
and an invalid or artifact-drifted catalog is blocked.

Filter the list down to one assay family, one lifecycle, or just broken configs:

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

Use explicit paths for scaffold/template configs when acting on them. Numeric
indexes only target the default `uv run reader ls` experiment list.

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
uv run reader dop classes
uv run reader dop classes --protocol plate_reader/retron_sponge_screen --format json
uv run reader dop ready-specs --format json
uv run reader notebook --list-templates
```

Scaffold a new experiment from a protocol:

```bash
uv run reader init ./experiments/20260317_new_assay --protocol <protocol-id>
```

Use `plate_reader/dual_reporter_screen` for CFP/YFP-style dual-reporter panels. Use `plate_reader/single_reporter_screen` for RFP-or-other single-reporter panels normalized to a configured denominator. Use `plate_reader/retron_sponge_screen` when the assay contract depends on matched same-sensor tetO controls plus compiled burden, leakiness, induced-effect, and cross-sensor ranking nodes.

For the matched-control sponge workflow itself, use the [Retron sponge screen guide](../guides/retron_sponge_screen.md). That guide maps the direct-ratio analysis sequence, the compiled assay tables, and the retron-specific plots and exports.

Inspect one experiment end to end:

```bash
uv run reader inspect CONFIG|DIR|INDEX
```

Emit the experiment as structured JSON with `authoring`, `semantics`, and
`implementation`:

```bash
uv run reader inspect CONFIG|DIR|INDEX --format json
uv run reader inspect CONFIG|DIR|INDEX --section readiness --format json
uv run reader inspect CONFIG|DIR|INDEX --section plan --format json
```

The stable `inspect --section` names are `identity`, `authoring`, `semantics`,
`plan`, `compiled`, `inputs`, `generated`, and `readiness`. Each projection
keeps experiment identity beside the requested section. Named protocol JSON
descriptions similarly accept `identity`, `authoring`, `semantics`, `defaults`,
or `compiled` through `protocols <id> --section NAME --format json`.

In JSON mode, `semantics.program` is the authored view of the active semantic
program for the experiment. The same program, with execution bindings and
coverage, lives under `implementation.compiled.semantic_program` beside the
compiled plugin wiring.

List just the pipeline chain and bindings:

```bash
uv run reader steps CONFIG|DIR|INDEX
uv run reader steps CONFIG|DIR|INDEX --format json
```

Guided walkthrough:

```bash
uv run reader demo
```

Protocol descriptions are the main place to check assay-specific inputs
and outputs. For the compact route, use [Common tasks](../guides/common_routes.md).
For machine-readable output, use [Automation and JSON](../guides/automation.md).

In short:

- `uv run reader protocols <id>` shows the protocol inputs, selected outputs, and compiled defaults.
- `uv run reader protocols <id> --example-config` prints a starter `reader/v8` outline.
- `uv run reader inspect`, `config`, `steps`, and `explain` show one bound experiment; JSON mode uses shared `authoring`, `semantics`, and `implementation` sections.
- `uv run reader ls --details --readiness` is the experiment list with preflight state.
- `uv run reader plot --list`, `uv run reader export --list`, and `uv run reader records` show selected outputs and generated records.
- `uv run reader plugins --protocol <id> --category ...` scopes registry inspection to the plugins a protocol uses by default.

---

## Configuration + validation

Inspect the experiment summary before reading the lower-level plan:

```bash
uv run reader inspect CONFIG|DIR|INDEX
```

The default table view includes a readiness panel so you can see, in one place,
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

In JSON mode, `authoring` is the full `reader/v8` document, while
`implementation` carries the compiled plan and the execution-bound semantic
program.

Validate schema, wiring, and inputs:

```bash
uv run reader validate CONFIG|DIR|INDEX
uv run reader validate CONFIG|DIR|INDEX --format json
```

On successful JSON validation, the envelope's `data` object keeps the preflight mode in `selection`,
then separates overall status/counts into `summary` from file-check details in
`validation`. `uv run reader validate --no-files --format json` still reports
declared file and auto-root counts even when the checks are skipped.

If you want the same preflight signal while browsing the whole experiment list, use
`uv run reader ls --details --readiness`. If you want the readiness view beside
one experiment’s compiled plan and current outputs, use `uv run reader inspect`.
For the full operating loop, use [Preflight, run, verify](../guides/preflight_run_verify.md).

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
uv run reader run CONFIG|DIR|INDEX --reset-records
```

Slice the pipeline:

```bash
uv run reader run CONFIG|DIR|INDEX --from step_a --until step_c
uv run reader run CONFIG|DIR|INDEX --only step_b
uv run reader run CONFIG|DIR|INDEX --from step_a --until step_c --dry-run --format json
```

`uv run reader run` fails fast if `--from` comes after `--until` in pipeline order.
Use `--reset-records` only to replace an incompatible generated catalog before
a complete pipeline rerun. It cannot be combined with a slice or dry run.

Inspect the emitted records catalog:

```bash
uv run reader records CONFIG|DIR|INDEX
uv run reader records CONFIG|DIR|INDEX --format json
uv run reader records CONFIG|DIR|INDEX --all --format json
```

In JSON mode, `uv run reader records` keeps experiment identity at the top of
`data`, then adds the record-manifest path, a summary by record kind and
producer, and the latest record entries. Current schema-v5 records bind the
complete config identity, Reader build identity, typed input evidence, exact
upstream revisions, and generated-file evidence. File bundles include one
typed description for every path. Non-v5 record payloads are rejected as an
invalid catalog and must be reproduced from source inputs. `--all` adds
revision counts rather than dumping every stored revision. Use `reader verify`
to prove the current catalog rather than treating `records` as an integrity
check.

Useful flags:

- `--from <step_id>` / `--until <step_id>` (pipeline only)
- `--only <step_id>` (single pipeline step)
- `--dry-run`
- `--reset-records` (complete pipeline rerun only)
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

For mutating runs, `reader plot --year` preflights the full batch first. If any
selected experiment is not runnable, the command aborts before writing plot
files so the year run does not leave partial state behind.

Override the experiments root when using `--year`:

```bash
uv run reader plot --year 2025 --root /path/to/experiments
```

List plot outputs and their upstream dataframe bindings:

```bash
uv run reader plot CONFIG|DIR|INDEX --list
uv run reader plot CONFIG|DIR|INDEX --list --format json
```

In JSON mode, `uv run reader plot --list` keeps the bound experiment at the top of `data`,
then adds `catalog`, `selection`, and `summary` blocks before the resolved
`plots` entries.

Dry-run a plot plan without executing:

```bash
uv run reader plot CONFIG|DIR|INDEX --dry-run
uv run reader plot CONFIG|DIR|INDEX --dry-run --format json
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

List exports and their upstream dataframe bindings:

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
uv run reader export CONFIG|DIR|INDEX --dry-run --format json
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

## Aggregate SFXI vec8

Render a cross-experiment heatmap from completed SFXI vec8 records or explicit
vec8 table files:

```bash
uv run reader init experiments/2026/20260708_sfxi_vec8_aggregate \
  --protocol workbench/generic \
  --title "SFXI vec8 cross-experiment aggregate"
uv run reader aggregate-sfxi-vec8 SOURCE... \
  --output-experiment experiments/2026/20260708_sfxi_vec8_aggregate
```

`SOURCE` may be an experiment config, experiment directory, outputs directory,
or a direct `.csv`, `.parquet`, or `.xlsx` vec8 table. Experiment and outputs
directory sources require the `sfxi_vec8/vec8` dataframe record. Pass
`outputs/exports/sfxi/vec8.xlsx` directly only when reviewing that exported
workbook snapshot.

Useful flags:

- `--output-experiment <CONFIG|DIR|INDEX>` is required. The named aggregate
  experiment owns the generated bundle through its configured `outputs/`
  directory; arbitrary workbench output directories are rejected.
- `--filename <name>` changes the artifact filename stem.
- `--title <text>` sets the heatmap title.
- `--dpi <n>` sets PNG resolution; the default is 300 DPI.
- `--overwrite` replaces an existing artifact bundle.
- `--format json` emits artifact paths, source rows, and summary counts.

The aggregate command writes a PNG heatmap, tidy CSV, and manifest. See
[SFXI vec8 in reader](../lib/sfxi_vec8_in_reader.md#aggregate-vec8-heatmap)
for the source and provenance rules.

---

## Notebooks

Scaffold a marimo notebook (no pipeline execution). Template selection is
ordered and protocol-constrained: explicit `--template`, then the first
compiled notebook spec from `config.yaml`, then the bound protocol default.
The selected template must be allowed by the protocol.

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
- `--mode run`: run as an app without the editor
- `--mode none`: create only (no launch)
- `--headless`: keep the server in the terminal and print a loopback URL for browser automation
- `--port <n>`: request a specific loopback port instead of the reader-managed clean-port selection

Runtime notes:

- `reader notebook` manages Marimo runtime state under `.cache/marimo/`.
- It reuses a live reader-managed session for the same notebook only when the notebook file and Reader runtime fingerprint match.
- If the notebook or runtime has drifted, it restarts the stale session instead of silently reusing it.
- It prunes older reader-managed sessions for the same experiment and launch mode before starting a new one.
- For agent review, prefer `--mode run --headless`, then open the printed URL in the in-app browser.
- Static HTML export can catch execution failures, but it does not validate live widget behavior. Use a served Marimo app for dropdown, slider, export-button, and chart-rerender checks.

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

List records from `outputs/manifests/records.json`:

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
