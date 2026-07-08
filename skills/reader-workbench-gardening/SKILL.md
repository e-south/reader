---
name: reader-workbench-gardening
description: Garden reader workbench architecture, docs, semantics, and verification surfaces. Use when auditing monolith pressure, assay lock-in, stale docs, CLI drift, or maintainer ergonomics. Do not use for experiment intake, generated-output edits, result interpretation, or branch/publish/CI work.
metadata:
  version: 0.3.1
  category: scientific-workbench
  tags: [reader, architecture, workbench, maintenance, harness, semantics]
---

# Reader Workbench Gardening

## Purpose

Keep `reader` easy to extend and operate by auditing information ownership,
reducing monolith pressure, and tightening maintainer-facing workbench
surfaces.

## Scope

In scope:

- workbench architecture and information-architecture audits
- assay lock-in, semantic drift, and stale-doc cleanup
- CLI, JSON, and preflight/run/verify surface hardening
- small, reversible maintainability improvements

Out of scope:

- new experiment intake or metadata staging
- hand-editing generated `outputs/`
- one-off feature delivery with no architecture or maintainability objective
- scientific interpretation disconnected from repo structure
- branch, publish, or CI topology work with no workbench-architecture objective

## Skill composition

- Pair with `deep-introspection` to map current boundaries, ownership, and
  runtime flow before changing them.
- Pair with `reader-data-operations-plan` when the architecture or docs pass is
  specifically about DOP policy, DOP registry coverage, or DOP intake
  contracts.
- Pair with `code-change-discipline` when boundary, contract, refactor, or
  fail-fast decisions need an explicit change strategy.
- Pair with `artifact-review-and-hardening` when the main deliverable is
  severity-ranked findings or maintainer hardening.
- Pair with `harness-engineering` when CLI, JSON, or end-to-end verification
  surfaces need stronger contracts or evidence.
- Pair with `evidence-writing` only when cleaning maintainer docs or skill prose
  after the technical content is settled.

## Inputs

- target surface or audit scope
- desired mode: audit-only, docs-sync, boundary-hardening, or
  surface-contracts
- known architectural pressure points or regressions
- representative experiment or protocol when runtime verification is needed
- explicit constraints: reliability, extensibility, delivery urgency, or
  publish boundaries

Clarification policy:

- ask only when missing context changes the target boundary or verification
  surface
- otherwise proceed with explicit assumptions and record them

## Success Criteria

- the cycle stays bounded to one ownership surface or one reversible slice
- findings and decisions are traceable to repository docs, code, or CLI
  evidence
- the chosen slice reduces coupling, drift, or retry cost without widening
  `reader/v7`
- verification matches the changed surface and skipped checks are explicit
- output separates verified facts, inferences, and follow-up work
- knowledge-integrity, autonomy-capability, and architecture-invariants checks
  remain satisfiable through deterministic repo-local commands

## Harness endpoints

Use [Endpoint contracts](./references/endpoint-contracts.md) to keep the skill
aligned with:

- `knowledge-integrity`
- `autonomy-capability`
- `architecture-invariants`

## Workflow

1. Treat
   [docs/guides/workbench_gardening.md](../../docs/guides/workbench_gardening.md)
   as the primary workflow. This skill stays thin and routes to that guide.
2. Use [Workflow reference](./references/workflow.md) to choose the mode and
   read order before making claims or edits.
3. If the task is really experiment intake, route to
   `reader-experiment-bootstrap`. If the task is really DOP classification,
   registry coverage, or DOP policy maintenance, route to
   `reader-data-operations-plan`. If branch state, publish flow, or CI topology
   becomes material, continue into
   [docs/repo-maintenance.md](../../docs/repo-maintenance.md).
4. Use [Checklists](./references/checklists.md) to look for monolith pressure,
   assay lock-in, stale semantics, doc drift, harness drift, and directory
   boundary violations.
5. Use [Endpoint contracts](./references/endpoint-contracts.md) to confirm
   which evidence surfaces must hold for this skill cycle.
6. Use [Verification](./references/verification.md) to choose the smallest
   verification bundle that matches the risk of the change.
7. Use [Test matrix](./references/test-matrix.md) for trigger checks,
   deterministic checks, and repeated-run consistency checks.
8. Use [External sources](./references/external-sources.md) when the cycle
   introduces claims that rely on tooling, standards, or behavior outside this
   repository.
9. If tracked docs or code changed, close through
   [docs/repo-change-gate.md](../../docs/repo-change-gate.md).

## Guardrails

- Do not restate `ARCHITECTURE.md` or `DESIGN.md` when the task only needs to
  point at their invariants.
- Do not treat a single assay family's needs as the workbench architecture.
- Do not hand-edit generated artifacts under `experiments/**/outputs/`.
- Do not widen public config or CLI surfaces when a narrower semantic move will
  solve the problem.
- Do not bundle unrelated refactors into the same gardening cycle.
- Do not let a gardening pass turn into generic repo-maintenance or release
  work unless the task explicitly expands.

## Required Deliverables

- scoped audit target, mode, and invariants
- endpoint contract coverage
- evidence summary: canonical docs, code surfaces, or CLI probes consulted
- ownership map or change target summary
- pressure or drift findings by category
- chosen smallest reversible slice or explicit audit-only recommendation
- verification evidence and skipped checks
- assumptions and residual risks

## Output Contract

Return:

1. Decision summary
   - target surface, mode, paired skills or docs used, invariants in scope, and
     selected endpoints
2. Evidence summary
   - canonical docs, code paths, and CLI probes used for this cycle
3. Endpoint coverage
   - how `knowledge-integrity`, `autonomy-capability`, and
     `architecture-invariants` were satisfied or deferred
4. Ownership and pressure summary
   - where meaning, mechanics, and docs currently live plus the drift or
     monolith signal
5. Selected slice
   - audit-only findings or the concrete change surface chosen, including why a
     broader slice was rejected
6. Verification bundle
   - commands run, CLI evidence, and skipped checks
7. Residual risks
   - deferred work, lock-in risk, and next maintenance pass

## Trigger Tests

Should trigger:

- "Audit `reader` for monolith pressure and assay lock-in."
- "Garden this workbench so it stays easy to extend."
- "Review the information architecture around protocols, plugins, and docs."
- "Tighten the CLI and verification surfaces after this maintainer cleanup."
- "Sync stale maintainer docs back to the current runtime and architecture."

Should not trigger:

- "Bootstrap a new experiment from these inputs."
- "Interpret the assay results."
- "Hand-edit the generated notebook outputs."
- "Add this feature with no architecture or maintainability objective."
- "Handle release, branch, or CI publish steps for this repo."

## Examples

Example 1: audit-only

- User says: "Audit the protocol and notebook surfaces for monolith pressure."
- Result: ownership map, pressure findings, and a smallest-next-slice
  recommendation with representative CLI evidence.

Example 2: docs-sync

- User says: "Bring the maintainer docs back in sync with current `reader/v7`
  behavior."
- Result: stale-route findings, canonical-doc fixes, and docs-only
  verification evidence.

Example 3: boundary-hardening

- User says: "Split this growing helper so a new assay family does not have to
  land in the same file."
- Result: one reversible boundary cut plus targeted verification and residual
  risk notes.

## Troubleshooting

- Scope keeps expanding:
  - narrow to one ownership boundary or one CLI/runtime surface
- Findings are correct but not actionable:
  - pick the smallest reversible slice and defer the rest
- The task overlaps adjacent routes:
  - hand experiment intake to `reader-experiment-bootstrap` and hand
    branch/publish or CI work to `docs/repo-maintenance.md`
- Verification is too broad:
  - use the smallest bundle from `references/verification.md`
- Docs start duplicating architecture prose:
  - point to the canonical document instead of restating it

## Additional resources

- [Workflow reference](./references/workflow.md)
- [Endpoint contracts](./references/endpoint-contracts.md)
- [Checklists](./references/checklists.md)
- [Verification](./references/verification.md)
- [Test matrix](./references/test-matrix.md)
- [External sources](./references/external-sources.md)
