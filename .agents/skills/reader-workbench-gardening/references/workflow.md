# Workflow reference

Use this reference while following
[docs/guides/workbench_gardening.md](../../../../docs/guides/workbench_gardening.md).

## Mode selection

- `audit-only`
  - map ownership, identify pressure or drift, and stop with a ranked next
    slice
- `docs-sync`
  - fix stale maintainer docs, routing, and command examples without changing
    runtime behavior
- `boundary-hardening`
  - make one small code or documentation cut that reduces concentration,
    coupling, or lock-in
- `surface-contracts`
  - tighten CLI, JSON, or preflight/run/verify evidence surfaces for agents and
    maintainers

If the task expands into branch state, publish flow, or CI topology, continue
into [docs/repo-maintenance.md](../../../../docs/repo-maintenance.md).

## Read order

1. [ARCHITECTURE.md](../../../../ARCHITECTURE.md)
2. [DESIGN.md](../../../../DESIGN.md)
3. [QUALITY.md](../../../../QUALITY.md) when the task changes quality or review
   expectations
4. [RELIABILITY.md](../../../../RELIABILITY.md) when the task changes preflight,
   run, verify, or recovery behavior
5. [docs/repo-maintenance.md](../../../../docs/repo-maintenance.md) when the task
   crosses repo boundaries, CI, or publish surfaces
6. [docs/repo-change-gate.md](../../../../docs/repo-change-gate.md) before
   finalizing tracked changes

## Minimal audit loop

1. State the workbench invariant or boundary under review.
2. Trace the surface through authored config, protocols, compiled declaration,
   runtime execution, and workbench outputs.
3. Use
   `uv run reader ls --root experiments --details --readiness --format json`
   when the cycle needs repo-wide discovery evidence.
4. Use
   `uv run reader inspect <config|dir|index> --format json` and
   `uv run reader explain <config|dir|index> --format json` when runtime
   mapping matters.
5. Record monolith pressure, assay lock-in, stale semantics, doc drift, or
   harness drift using [checklists.md](./checklists.md).
6. Confirm which endpoint contracts matter using
   [endpoint-contracts.md](./endpoint-contracts.md).
7. Choose the smallest reversible slice that improves ownership clarity or
   verification discipline.
8. Run the smallest matching verification bundle from
   [verification.md](./verification.md).

## Default commands

```bash
uv run reader ls --root experiments --details --readiness --format json
uv run reader inspect <config|dir|index> --format json
uv run reader explain <config|dir|index> --format json
uv run reader validate <config|dir|index> --no-files --format json
uv run reader run <config|dir|index> --dry-run --format json
```

Add `plot --list`, `export --list`, `records`, or a real execution slice only
when the gardening cycle changes those surfaces.
