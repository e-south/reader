---
doc_id: reader-experiment-bootstrap
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-07-28
summary: Workflow for creating an experiment workspace from classified inputs and verifying the first Reader run.
---

# Experiment Bootstrap

Use this guide when creating a new `reader` experiment from raw assay data or
when auditing the local experiment list. This is the main creation and intake workflow
for the recurring "classify the data, find a similar experiment, materialize
inputs, wire config, build metadata, preflight, run, verify" loop.

## Principles

- Keep `AGENTS.md` as the map, not the encyclopedia.
- Start with the [Data Operations Plan](./data_operations_plan.md) data class
  before copying templates or authoring config.
- Prefer existing protocol contracts and nearby experiment templates over
  bespoke config authoring.
- Do not hand-edit generated `outputs/`; regenerate instead.
- Do not silently infer plate semantics when ambiguity changes well identity,
  treatment meaning, or control interpretation.
- Treat tracked repo fixtures and local experiments as different audit
  scopes. CI should stay stable; local experiment list audits should include ignored
  experiments.

## 1. Classify the data class

Start by selecting the first matching class from
[Data classes](./data_operations_plan/data_classes.md):

- plate-reader screen
- flow-cytometry panel
- logic/SFXI analysis
- aggregate/review workspace
- unsupported long-tail assay

The selected class should determine the preferred protocol family, metadata
minimums, and transfer expectations. If no class fits, keep the experiment as
`draft` or `template` and document the missing protocol/metadata contract
instead of forcing the data into a nearby protocol.

## 2. Discover the assay family and nearest template

Start with the local experiment list:

```bash
uv run reader ls --root experiments --details --readiness
uv run reader ls --root experiments --details --readiness --format json
```

Prefer JSON when another tool or agent will consume the output.

Then narrow to the assay family you need:

```bash
uv run reader ls --root experiments --details --protocol plate_reader/single_reporter_screen
uv run reader ls --root experiments --details --protocol plate_reader/dual_reporter_screen
uv run reader protocols <protocol-id>
uv run reader protocols <protocol-id> --example-config
```

Pick the closest prior experiment by:

- data class
- protocol id
- raw instrument family
- channel semantics
- metadata shape
- plot and export choices

Inspect before copying:

```bash
uv run reader inspect <config|dir|index>
uv run reader steps <config|dir|index>
uv run reader explain <config|dir|index>
```

## 3. Create the workspace

Use `reader init` when protocol defaults are the main starting point:

```bash
uv run reader init ./experiments/YYYY/YYYYMMDD_shortslug --protocol <protocol-id>
```

Use a nearest-neighbor config when the new run is semantically close to a prior
experiment and you need to preserve annotations, plot ids, or export behavior.

Either way, keep the standard layout:

```text
experiments/YYYY/YYYYMMDD_shortslug/
  config.yaml
  inputs/
  notebooks/
```

Keep hand-authored notebooks in `notebooks/`. `reader notebook` writes generated
scaffolds under `outputs/notebooks/`. Reader creates `outputs/` lazily when a
command produces generated state.

## 4. Intake raw data

Use the raw workbook or instrument export as the source of truth and keep the
original filename in `inputs/`.

If the source is Google Drive or another external system, use the configured
workspace integration to materialize the original file, then record the source
identity and staged SHA-256 in the intake note. Reader begins at the local
experiment boundary; it does not own remote transfer semantics.

When the workbook schema drifts from prior experiments, inspect it before
editing config:

- sheet names
- channel labels
- whether the file is kinetic-only or multi-part
- whether the data is a native workbook or an imported Google file

For `mixed` or `snapshot_only` Synergy parsing, declare
`protocol.inputs.ingest.channel_map` from workbook labels to canonical channel
names. Reader validates snapshot labels against this map; list order does not
assign channel identity. Matching preserves the declared wavelength suffix and
is exact after whitespace normalization and removal of BioTek's anchored
one-letter `A` or `B` block suffix immediately before `:`. In `mixed` mode,
every mapped channel must occur in both snapshot and kinetic data; one source
cannot hide missing measurements in the other. Auto-discovery accepts one
modern `.xlsx` workbook by default. Use `auto_pick: latest` only when selecting
the newest file is an intentional experiment policy.

## 5. Build metadata deliberately

Use the nearest prior metadata workbook or CSV as the formatting template, but
rewrite the semantic content for the new experiment.

Preserve these contracts:

- Every measured position must be accounted for.
- If the sample-map workflow expects full plate coverage, keep every well in the
  metadata table and leave truly unused wells metadata-empty rather than
  deleting rows.
- Keep blanks explicit only when the assay semantics actually require them.
- Preserve workbook structure when downstream parsing depends on it.

Ask the user for missing or conflicting metadata when any of these are unclear:

- well coordinates
- design ids / strain ids
- treatment lattice
- blank/control interpretation
- desired alias labels

Do not silently resolve collisions like overlapping well assignments. Surface
the ambiguity and get confirmation first.

## 6. Preflight the smallest slice first

Use the normal `reader` loop:

```bash
uv run reader validate <config|dir|index> --no-files
uv run reader validate <config|dir|index>
uv run reader run <config|dir|index> --dry-run --format json
uv run reader plot <config|dir|index> --list
uv run reader export <config|dir|index> --list
```

Use the cheapest command that answers the next question:

- config shape only: `validate --no-files`
- file presence and dependency readiness: `validate`
- compiled execution slice: `run --dry-run`
- output portfolio: `plot --list` / `export --list`

## 7. Execute and verify

Run only after preflight is clean:

```bash
uv run reader run <config|dir|index>
uv run reader plot <config|dir|index>
uv run reader export <config|dir|index>
uv run reader records <config|dir|index>
uv run reader verify <config|dir|index>
```

Verification should include:

- `outputs/manifests/records.json`
- a successful `reader verify` result
- expected plot files
- expected export files
- any key fold-change or summary tables the experiment is supposed to produce

## 8. Audit the local experiment list

The repo test suite only covers tracked fixture experiments. To audit the real
local experiment directories under `experiments/`, use the local audit tool.
Omit `--years` to audit every numeric year directory under `experiments/`, or
pass explicit years when you want a narrower run:

```bash
uv run reader audit experiments
uv run reader audit experiments --format json
uv run reader audit experiments --years <yyyy> [--years <yyyy>]
uv run reader audit experiments --include-non-active
```

This tool stages experiments into temporary copies so the audit does not mutate
the original experiment outputs. By default it skips non-active lifecycles and
reports them separately. Use `--include-non-active` only when you intentionally
want to pressure-test draft/template configs too.

## Common friction

- The local experiment list is broader than the tracked repo fixture set.
- Workbook channel names drift more often than protocol ids do.
- Sample-map failures usually come from incomplete plate coverage, not parser
  bugs.
- Draft experiments should not be forced through an end-to-end run.
- Google Drive materialization is external state; keep that step explicit in the
  audit output or final handoff.
