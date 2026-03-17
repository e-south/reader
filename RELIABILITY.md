# Reliability

`reader` should be reliable in the operational sense, not just the implementation sense. A reliable workbench lets a user or agent predict what will run, verify inputs before mutation, and recover cleanly when something is wrong.

This document describes the expected reliability loop.

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
- Removed legacy keys fail fast.
- Pipeline slicing is explicit.
- Input/output ports are typed.
- Records are persisted with provenance and content digests.
- Plotting cache setup is explicit and writable.
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
- inputs
- config digest
- content digest
- optional source-recipe provenance

That provenance is what lets the workbench stay inspectable as experiments and plots accumulate.

## Throughput And Feedback Latency

Fast feedback is a reliability feature.

Use the cheapest command that answers the current question.

- use `reader validate --no-files` for schema-only checks
- use `reader run --dry-run` before executing a slice
- use `reader ls --details --format json` for fleet-wide inspection
- use `reader inspect --format json` for one experiment’s current state

This follows the same harness principle highlighted in OpenAI’s harness-engineering article: better harnesses reduce retries by shortening the path from action to trustworthy feedback.

## Reliability Invariants

- Discovery and preflight should not require full execution.
- Machine-readable contracts should not silently fall back to human-only output.
- Generated outputs should be reproducible from config and code.
- Records should remain the authoritative provenance surface.
- Experiment directories should remain compact and legible.

## Current Reliability Debt

The largest remaining reliability debt is semantic rather than operational. Protocol controls, windows, metrics, and ranking are still not one executable typed analysis DAG. That means some assay truth is still split between protocol metadata and compiler behavior.

Operationally, the workbench is more reliable when inspection and dry-run routes are used first. Architecturally, full reliability requires finishing that semantic cut.

## Related Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [DESIGN.md](./DESIGN.md)
- [QUALITY.md](./QUALITY.md)
- [SECURITY.md](./SECURITY.md)
