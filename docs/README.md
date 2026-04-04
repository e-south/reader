# Documentation index

Use this index to find the smallest document that matches what you need. Start
with the guides if you want to understand how `reader` is organized or how an
experiment moves from inputs to outputs. Use the reference pages when you need
exact CLI or config details.

## Start here

- [Getting started](./guides/getting_started.md): install `reader`, check the environment, and inspect the first experiment.
- [Common tasks](./guides/common_routes.md): shortest command paths for discovery, validation, execution, and JSON output.

## Core workflows

- [Preflight, run, verify](./guides/preflight_run_verify.md): deterministic operating path for one experiment.
- [Automation and JSON](./guides/automation.md): machine-readable discovery, inspection, and preflight surfaces.
- [End-to-end demo](./guides/demo.md): one concrete walkthrough from discovery to outputs.

## User guides

- [Retron sponge screen guide](./guides/retron_sponge_screen.md): matched-control sponge assay setup, runtime flow, plots, and exports.
- [Notebooks](./guides/notebooks.md): notebook scaffolding and Marimo usage in experiment directories.
- [Marimo reference](./guides/marimo_reference.md): notebook widgets, patterns, and examples.

## Reference

- [CLI reference](./core/cli.md): full command reference.
- [Configuring `reader/v7`](./core/pipeline.md): config schema and protocol-owned authoring surface.

## Maintainer docs

- [Repo change gate](./repo-change-gate.md): minimum gate before landing tracked changes.
- [Repo maintenance](./repo-maintenance.md): repo-wide verification, CI lanes, and maintenance surfaces.
- [Plugin development](./core/plugins.md): add or extend ingest, transform, plot, export, and validator plugins.
- [Architecture](../ARCHITECTURE.md): system structure, ownership boundaries, and invariants.
- [Design](../DESIGN.md): product and information-design rules for the public surface.
- [Quality](../QUALITY.md): quality bar, evidence expectations, and failure taxonomy.
- [Reliability](../RELIABILITY.md): preflight, run, verify, and recovery expectations.
- [Security](../SECURITY.md): trust boundaries and safe defaults.
- [Spec / architecture](./core/spec.md): deeper package layout and implementation notes.
- [Dev journal](./dev/journal.md): change log for major design and architecture cuts.

## Library notes

- [Crosstalk pairs](./lib/crosstalk_pairs.md)
- [SFXI vec8 in reader](./lib/sfxi_vec8_in_reader.md)
