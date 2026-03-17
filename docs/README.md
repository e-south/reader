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
  - `reader ls --details --protocol <protocol-id>`
  - `reader ls --details --status config_error`
  - `reader inspect <config|dir|index>`
  - `reader explain <config|dir|index>`
- I need machine-readable discovery for an agent or automation:
  - `reader ls --details --format json`
  - `reader ls --details --protocol <protocol-id> --format json`
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
  - `reader ls` JSON includes a fleet summary by protocol/status/output state.
  - `reader protocols` / `inspect` / `explain` JSON include a protocol semantic program with explicit `compiled` vs `descriptive_only` execution status for controls, windows, metrics, and ranking.
  - `reader steps` / `inspect` / `plot --list` / `export --list` JSON include upstream producer and contract-surface metadata for record bindings.
- I need to discover available assays and their outputs:
  - `reader protocols`
  - `reader protocols <protocol-id>`
  - `reader protocols <protocol-id> --example-config`
  - `reader plugins --protocol <protocol-id> --category transform`
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
