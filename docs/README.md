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
  - `uv run reader ls --details`
  - `uv run reader ls --details --readiness`
  - `uv run reader ls --details --lifecycle draft`
  - `uv run reader ls --details --protocol <protocol-id>`
  - `uv run reader ls --details --status config_error`
  - `uv run reader inspect <config|dir|index>`
  - `uv run reader explain <config|dir|index>`
- I need machine-readable discovery for an agent or automation:
  - `uv run reader ls --details --format json`
  - `uv run reader ls --details --readiness --format json`
  - `uv run reader ls --details --protocol <protocol-id> --format json`
  - `uv run reader config <config|dir|index> --format json`
  - `uv run reader inspect <config|dir|index> --format json`
  - `uv run reader steps <config|dir|index> --format json`
  - `uv run reader validate <config|dir|index> --no-files --format json`
  - `uv run reader explain <config|dir|index> --format json`
  - `uv run reader run <config|dir|index> --dry-run --format json`
  - `uv run reader protocols <protocol-id> --format json`
  - `uv run reader plugins --protocol <protocol-id> --format json`
  - `uv run reader plot <config|dir|index> --list --format json`
  - `uv run reader export <config|dir|index> --list --format json`
  - `uv run reader records <config|dir|index> --format json`
  - `uv run reader ls` JSON uses `catalog`, `selection`, `summary`, and `experiments`, with fleet totals grouped by protocol, config status, lifecycle, and output state.
  - `uv run reader ls --details --readiness` adds per-experiment readiness plus fleet totals grouped by readiness state.
  - `uv run reader protocols`, `uv run reader config`, `uv run reader steps`, `uv run reader inspect`, and `uv run reader explain` JSON all use shared layered sections: `authoring`, `semantics`, and `implementation`, plus command-specific envelope fields.
  - `uv run reader records` JSON carries experiment identity, the manifest path, summary counts by record kind/producer, and optional revision counts with `--all`.
  - `uv run reader plugins` JSON keeps registry filters under `selection` and adds ontology summaries by category, domain, and family.
  - `semantics.program` carries explicit `compiled` vs `descriptive_only` execution status for controls, windows, metrics, and ranking.
  - `semantics.program.summary` gives fast coverage counts so agents can see whether a protocol is mostly executable or still largely descriptive.
  - `uv run reader config --format json` keeps the full `reader/v7` document under `authoring` and the compiled runtime chain under `implementation`.
  - `uv run reader validate --format json` keeps preflight mode under `selection`, summary totals under `summary`, and file-check details under `validation`.
  - `uv run reader validate --no-files --format json` still reports declared file and auto-root counts even when checks are skipped.
  - `uv run reader inspect --format json` carries the same readiness view under `implementation.readiness` so agents can tell whether a config is draft/template, blocked by dependencies or files, runnable, already has records, or only has legacy outputs without composing extra calls.
  - `uv run reader inspect` / `uv run reader explain` keep runtime filesystem state under `implementation`, not mixed into top-level assay semantics.
  - `uv run reader steps` / `inspect` / `plot --list` / `export --list` JSON include upstream producer and contract-surface metadata for record bindings.
  - `uv run reader plot --list` / `uv run reader export --list` JSON keep user filters under `selection` and add compact output summaries by plugin, domain, and family.
- I need to discover available assays and their outputs:
  - `uv run reader protocols`
  - `uv run reader protocols <protocol-id>`
  - `uv run reader protocols <protocol-id> --example-config`
  - `uv run reader plugins --protocol <protocol-id> --category transform`
  - `uv run reader protocols plate_reader/retron_sponge_screen` for matched-control sponge screens
  - `uv run reader protocols plate_reader/dual_reporter_screen` for general dual-reporter sensor panels
- I need to scaffold a new experiment from an assay:
  - `uv run reader init ./experiments/<new-experiment> --protocol <protocol-id>`
  - `uv run reader inspect ./experiments/<new-experiment>/config.yaml`
- I need to see what plots or artifacts a config will generate:
  - `uv run reader plot <config|dir|index> --list`
  - `uv run reader plot <config|dir|index> --list --format json`
  - `uv run reader export <config|dir|index> --list`
  - `uv run reader export <config|dir|index> --list --format json`
- I need to add or extend a maintainer surface:
  - read [Plugin development](./core/plugins.md)
  - then read [Spec / architecture](./core/spec.md)
