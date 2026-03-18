# reader documentation

`reader` is an experimental workbench. The docs are organized for progressive disclosure: start with the user-facing workflow, then drop into maintainer or library details only when you need them.

## Start here

- [CLI reference](./core/cli.md): discover assays and experiments, scaffold a new experiment, inspect plans, validate configs, run pipelines, list plots/exports, and scaffold notebooks.
- [Configuring reader/v7](./core/pipeline.md): the canonical YAML authoring surface.
- [End-to-end demo](./guides/demo.md): a concrete walk-through of the normal workbench flow.

## User docs

- [CLI reference](./core/cli.md)
- [Configuring reader/v7](./core/pipeline.md)
- [Notebooks](./guides/notebooks.md)
- [Marimo reference](./guides/marimo_reference.md)

## Maintainer docs

- [Repo change gate](./repo-change-gate.md)
- [Repo maintenance](./repo-maintenance.md)
- [Architecture](../ARCHITECTURE.md)
- [Design](../DESIGN.md)
- [Quality](../QUALITY.md)
- [Reliability](../RELIABILITY.md)
- [Security](../SECURITY.md)
- [Plugin development](./core/plugins.md)
- [Spec / architecture](./core/spec.md)
- [Dev journal](./dev/journal.md)

## Demos and library notes

- [End-to-end demo](./guides/demo.md)
- [Crosstalk pairs](./lib/crosstalk_pairs.md)
- [SFXI vec8 in reader](./lib/sfxi_vec8_in_reader.md)

## Common routes

- I need to understand what an experiment does:
  - `reader ls --details`
  - `reader ls --details --readiness`
  - `reader ls --details --protocol <protocol-id>`
  - `reader ls --details --status config_error`
  - `reader inspect <config|dir|index>`
  - `reader explain <config|dir|index>`
- I need machine-readable discovery for an agent or automation:
  - `reader ls --details --format json`
  - `reader ls --details --readiness --format json`
  - `reader ls --details --protocol <protocol-id> --format json`
  - `reader config <config|dir|index> --format json`
  - `reader inspect <config|dir|index> --format json`
  - `reader steps <config|dir|index> --format json`
  - `reader validate <config|dir|index> --no-files --format json`
  - `reader explain <config|dir|index> --format json`
  - `reader run <config|dir|index> --dry-run --format json`
  - `reader protocols <protocol-id> --format json`
  - `reader plugins --protocol <protocol-id> --format json`
  - `reader plot <config|dir|index> --list --format json`
  - `reader export <config|dir|index> --list --format json`
  - `reader records <config|dir|index> --format json`
  - `reader ls` JSON uses `catalog`, `selection`, `summary`, and `experiments`, with fleet totals grouped by protocol, status, and output state.
  - `reader ls --details --readiness` adds per-experiment readiness plus fleet totals grouped by readiness state.
  - `reader protocols`, `reader config`, `reader steps`, `reader inspect`, and `reader explain` JSON all use the same top-level layers: `authoring`, `semantics`, and `implementation`.
  - `reader records` JSON carries experiment identity, the manifest path, summary counts by record kind/producer, and optional revision counts with `--all`.
  - `reader plugins` JSON keeps registry filters under `selection` and adds ontology summaries by category, domain, and family.
  - `semantics.program` carries explicit `compiled` vs `descriptive_only` execution status for controls, windows, metrics, and ranking.
  - `semantics.program.summary` gives fast coverage counts so agents can see whether a protocol is mostly executable or still largely descriptive.
  - `reader config --format json` keeps the full `reader/v7` document under `authoring` and the compiled runtime chain under `implementation`.
  - `reader validate --format json` keeps preflight mode under `selection`, summary totals under `summary`, and file-check details under `validation`.
  - `reader inspect --format json` carries the same readiness view under `implementation.readiness` so agents can tell whether a config is blocked, runnable, or already has records without composing extra calls.
  - `reader inspect` / `reader explain` keep runtime filesystem state under `implementation`, not mixed into top-level assay semantics.
  - `reader steps` / `inspect` / `plot --list` / `export --list` JSON include upstream producer and contract-surface metadata for record bindings.
  - `reader plot --list` / `reader export --list` JSON keep user filters under `selection` and add compact output summaries by plugin, domain, and family.
- I need to discover available assays and their outputs:
  - `reader protocols`
  - `reader protocols <protocol-id>`
  - `reader protocols <protocol-id> --example-config`
  - `reader plugins --protocol <protocol-id> --category transform`
  - `reader protocols plate_reader/retron_sponge_screen` for matched-control sponge screens
  - `reader protocols plate_reader/dual_reporter_screen` for general dual-reporter sensor panels
- I need to scaffold a new experiment from an assay:
  - `reader init ./experiments/<new-experiment> --protocol <protocol-id>`
  - `reader inspect ./experiments/<new-experiment>/config.yaml`
- I need to see what plots or artifacts a config will generate:
  - `reader plot <config|dir|index> --list`
  - `reader plot <config|dir|index> --list --format json`
  - `reader export <config|dir|index> --list`
  - `reader export <config|dir|index> --list --format json`
- I need to add or extend a maintainer surface:
  - read [Plugin development](./core/plugins.md)
  - then read [Spec / architecture](./core/spec.md)
