---
doc_id: reader-reliability
surface: reliability-contract
owner: reader-maintainers
last_verified: 2026-07-11
summary: Operational contract for Reader discovery, preflight, execution, provenance checks, and recovery.
---

# Reliability

`reader` should be reliable in the operational sense, not just the implementation sense. A reliable workbench lets a user or agent predict what will run, verify inputs before mutation, and recover cleanly when something is wrong.

This document describes the expected reliability loop.

For the task-oriented route, use [docs/guides/preflight_run_verify.md](./docs/guides/preflight_run_verify.md).

## Reliability Contract

The canonical operating path is:

1. discover
2. inspect
3. validate
4. dry-run
5. execute
6. verify outputs and provenance

`reader` should make each phase explicit through the CLI.

## Preflight, Run, Verify

### Preflight

Use these commands before mutating state:

- `reader inspect`
- `reader steps`
- `reader validate`
- `reader explain`
- `reader run --dry-run`
- `reader plot --list`
- `reader export --list`

These commands answer different reliability questions.

- `inspect`
  What is this experiment bound to do?
- `steps`
  What is the pipeline daisy chain?
- `validate`
  Is the config wired correctly?
- `explain`
  What full compiled plan will the runtime use?
- `run --dry-run`
  What pipeline slice would execute?
- `plot/export --list`
  What semantic outputs resolve from this config?

### Run

Mutation happens only in the execution commands:

- `reader run`
- `reader plot`
- `reader export`
- `reader notebook`

### Verify

After execution, verify:

- `outputs/manifests/records.json`
- generated plot and export files
- `reader records`
- `reader inspect`

The verify phase should prove what happened, not ask the operator to infer it from filesystem sprawl.

## Deterministic Path Expectations

`reader` reliability depends on the following properties.

- Config schema is strict.
- Removed config keys fail fast.
- Pipeline slicing is explicit.
- Input/output ports are typed.
- Plot and export steps return explicit nonempty file outputs.
- Dataframe records are persisted with provenance and verified content
  digests; file bundles are path-confined and described.
- Figure rendering does not depend on process-global object-identity caches.
- JSON inspection surfaces are stable enough for automation.

## Recovery And Failure Handling

The preferred failure mode is explicit refusal, not silent repair.

Examples:

- broken config -> `reader validate` or `reader ls` surfaces a config error
- invalid reads/writes linkage -> validation error before execution
- missing records catalog -> explicit `reader records` failure
- invalid JSON mode for mutating commands -> hard parameter error

Recovery should follow the shortest deterministic path.

1. fix authored config or code
2. rerun preflight
3. rerun only the affected execution slice
4. verify the updated records and outputs

## Provenance Expectations

Reliability is incomplete without provenance.

`reader` persists:

- latest records
- record history
- producer identity
- producer plugin
- runtime input references
- effective plugin-config digest
- dataframe artifact content digest
- optional source-recipe provenance

Raw file references do not yet carry source-content digests, and file bundles
do not yet carry per-file content digests. Those gaps must remain visible in
inspection and documentation until a stricter record schema supplies them.

The current provenance keeps the workbench inspectable as experiments and
plots accumulate without implying stronger source integrity than the catalog
actually records.

Experiment-scoped retron reviews resolve semantic summary and trace inputs from
the source record catalog, require the current dataframe contracts, and verify
artifact content digests. A nearby CSV export is not a substitute for the
cataloged record.

## Throughput And Feedback Latency

Fast feedback is a reliability feature.

Use the cheapest command that answers the current question.

- use `reader validate --no-files` for schema-only checks
- use `reader run --dry-run` before executing a slice
- use `reader ls --details --format json` for workbench-wide inspection
- use `reader inspect --format json` for one experiment’s current state

Shorter feedback loops reduce retries by shortening the path from action to trustworthy feedback.

Use [docs/guides/automation.md](./docs/guides/automation.md) for the compact JSON route.

## Reliability Invariants

- Discovery and preflight should not require full execution.
- Machine-readable contracts should not silently fall back to table output.
- Generated outputs should be reproducible from config and code.
- Records should remain the authoritative provenance surface.
- New plot records require complete, duplicate-free path descriptions from an
  exact protocol figure or explicit producer metadata.
- Experiment directories should remain compact and legible.

## Current Reliability Debt

- Large protocol and retron presentation modules raise the cost of reviewing a
  local change and make incomplete test selection more likely.
- Files representing formats or partitions of one protocol figure share its
  figure-level summary. Multi-panel figures that need panel-specific accessible
  descriptions require a separate presentation-layer contract.
- A Marimo syntax check does not exercise reactive browser interactions; a
  representative live notebook review remains necessary for UI changes.

## Related Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [DESIGN.md](./DESIGN.md)
- [QUALITY.md](./QUALITY.md)
- [SECURITY.md](./SECURITY.md)
