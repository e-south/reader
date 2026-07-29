# Endpoint Contracts

Use these endpoint contracts when hardening or validating
`reader-data-operations-plan`.

## `knowledge-integrity`

Goal:
- Keep DOP policy discoverable, source-backed, and cross-linked to owned repo
  surfaces.

Required evidence:

- [Data Operations Plan](../../../docs/guides/data_operations_plan.md) remains
  the primary DOP overview
- [Operating model](../../../docs/guides/data_operations_plan/operating_model.md)
  owns DOP component boundaries
- [External sources](./external-sources.md) records dated source rows for
  external DOP claims
- `uv run reader maintain docs` passes after docs or route edits

Failure handling:
- Repair missing routes before changing prose. If the source claim is stale or
  unavailable, mark the claim as unverified instead of strengthening it.

## `autonomy-capability`

Goal:
- Let agents classify DOP data classes and check ready gates without scraping
  prose or guessing from prior memory.

Required evidence:

- `uv run reader dop classes --format json` works for data-class facts
- `uv run reader dop ready-specs --format json` works for evidence gates
- [Test matrix](./test-matrix.md) names trigger, functional, and deterministic
  checks
- `uv run reader maintain skills` passes after skill edits

Failure handling:
- If JSON registry output is unavailable, stop the automation-facing claim and
  report the registry check failure. Do not replace registry facts with prose
  inference.

## `architecture-invariants`

Goal:
- Keep DOP policy decoupled from experiment execution and public config schema.

Required evidence:

- The skill routes full experiment creation to `reader-experiment-bootstrap`
- The skill routes broad architecture maintenance to
  `reader-workbench-gardening`
- The DOP registry remains read-only guidance unless a separate code-change
  contract changes execution behavior
- Generated `experiments/**/outputs/` stay out of scope

Failure handling:
- Stop when a DOP update would widen `reader/v8`, hand-edit generated outputs,
  or encode guessed metadata semantics. Route to the owning workflow instead.
