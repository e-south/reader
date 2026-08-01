---
doc_id: reader-docs-index
surface: documentation-index
owner: reader-maintainers
last_verified: 2026-08-01
summary: Routes readers to the smallest current guide or reference for operating and maintaining reader.
---

# Documentation index

Use this index to find the smallest doc that answers the current question.
Start with the guides if you want to see how `reader` moves from inputs to
outputs. Use the reference pages when you need exact CLI or config details.

## Start here

- [Getting started](./guides/getting_started.md): install `reader`, check the
  environment, and inspect a first experiment.
- [Common tasks](./guides/common_routes.md): shortest routes for discovery,
  validation, execution, and JSON output.

## Core workflows

- [Preflight, run, verify](./guides/preflight_run_verify.md): inspect,
  validate, and execute one experiment.
- [Automation and JSON](./guides/automation.md): machine-readable discovery,
  inspection, and preflight routes.
- [Data Operations Plan](./guides/data_operations_plan.md): classify datasets,
  capture metadata minimums, and keep intake decisions explicit.
- [Experiment bootstrap](./guides/experiment_bootstrap.md): create an
  experiment from local or Drive-backed inputs and verify the run.
- [End-to-end demo](./guides/demo.md): one concrete walkthrough from discovery to outputs.

## User guides

- [Notebooks](./guides/notebooks.md): notebook scaffolding and Marimo usage in
  experiment directories.
- [Marimo reference](./guides/marimo_reference.md): notebook widgets, patterns, and examples.

## Reference

- [CLI reference](./core/cli.md): full command reference.
- [Python API](./core/python_api.md): typed inspection, planning, record, and
  verification entrypoints for integrations.
- [Record provenance](./core/record_provenance.md): catalog-v4 epochs,
  record-schema-v6 evidence, and verification states.
- [Configuring `reader/v8`](./core/pipeline.md): config schema and protocol-owned settings.
- [Ordered state spaces](./core/ordered_state_spaces.md): metric-neutral state
  identity and exact metadata-value binding.

## Maintainer docs

- [Repo change gate](./repo-change-gate.md): minimum gate before landing
  tracked changes.
- [Repo maintenance](./repo-maintenance.md): repo-wide checks, CI, and
  maintenance guidance.
- [Workbench gardening](./guides/workbench_gardening.md): maintainer workflow
  for architecture, docs, and verification-surface cleanup.
- [Plugin development](./core/plugins.md): add or extend ingest, transform,
  plot, export, and validator plugins.
- [Architecture](../ARCHITECTURE.md): system structure, ownership boundaries, and invariants.
- [Design](../DESIGN.md): product and information-design rules for the public
  UI and docs.
- [Quality](../QUALITY.md): quality bar, evidence expectations, and failure taxonomy.
- [Reliability](../RELIABILITY.md): preflight, run, verify, and recovery expectations.
- [Security](../SECURITY.md): trust boundaries and safe defaults.
- [Spec / architecture](./core/spec.md): deeper package layout and implementation notes.

## Library notes

- [Cytometry flow panels](./lib/cytometry.md): explicit FCS ingestion, gating,
  typed statistics, QC, registered diagnostics, and exports.
- [Crosstalk pairs](./lib/crosstalk_pairs.md)
- [Logic-symmetry space](./lib/logic_symmetry.md): generic four-state geometry,
  records, configuration, and plot semantics.
- [Plate-reader metric outputs](./lib/plate_reader/metric_outputs.md): shared
  plate-reader records and protocol-owned four-state vector and response-window analyses.
- [Four-state vector in Reader](./lib/four_state_vector_in_reader.md): measured vector
  generation, workflow, and plot surfaces.
- [Plate-reader response-window analysis](./lib/plate_reader/response_window.md):
  event-relative summaries from declared cross-experiment record resources.
