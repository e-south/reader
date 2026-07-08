---
doc_id: reader-workbench-gardening
surface: maintainer-guide
owner: reader-maintainers
last_verified: 2026-07-08
summary: Maintainer workflow for bounded reader architecture, documentation, CLI, and verification gardening.
---

# Workbench gardening

Use this guide when the task is to keep `reader` easy to change, assay
extensible, and operationally clear for maintainers. The matching repo-local
skill routes here; this document is the primary workflow.

## When to use this

Use this guide for:

- architecture and information-architecture audits
- semantic monolith pressure around protocols, compiler surfaces, notebooks, or
  registries
- assay lock-in risk in config, CLI, docs, or code organization
- stale semantics, stale docs, or legacy behavior that no longer matches
  `reader/v7`
- maintainer ergonomics and CLI or JSON surface hardening

Do not use this guide for new-experiment intake, result interpretation, or
hand-editing generated outputs under `experiments/**/outputs/`.

## Adjacent routes

Use a narrower or broader route when the task is not primarily about workbench
architecture or maintainability:

- use [Experiment bootstrap](./experiment_bootstrap.md) for new-experiment
  intake, metadata staging, or local experiment audits
- use [Repo maintenance](../repo-maintenance.md) when branch state, publish
  flow, or CI topology becomes part of the task
- use [Plugin development](../core/plugins.md) when the real work is adding a
  new plugin mechanic rather than reducing workbench drift

## Gardening modes

Pick one mode before you start so the cycle stays reviewable:

- `audit-only`
  - map ownership and pressure, then stop with a ranked next slice
- `docs-sync`
  - bring maintainer docs and routes back in sync with actual runtime behavior
- `boundary-hardening`
  - make one small structural cut that reduces concentration or lock-in
- `surface-contracts`
  - tighten CLI, JSON, or preflight/run/verify evidence surfaces

## Core invariants

Start every gardening cycle by checking the current invariants instead of
rephrasing them from memory:

- [ARCHITECTURE.md](../../ARCHITECTURE.md)
- [DESIGN.md](../../DESIGN.md)
- [QUALITY.md](../../QUALITY.md)
- [RELIABILITY.md](../../RELIABILITY.md)
- [docs/repo-change-gate.md](../repo-change-gate.md)

The invariants that usually matter most are:

- experiment-scoped IO remains the workbench unit of work
- `reader/v7` stays the only public config schema
- protocols own assay semantics and output vocabulary
- plugins stay mechanical adapters around domain or runtime logic
- discovery, validation, and dry-run surfaces stay first-class
- generated outputs remain generated and manifest-backed

## Skill composition

Pair this maintainer workflow with global skills when the cycle needs a deeper
specialized lens:

- `deep-introspection` to map current architecture, ownership, and runtime flow
- `code-change-discipline` to choose boundaries, contracts, refactor strategy,
  and fail-fast behavior
- `artifact-review-and-hardening` when the main deliverable is findings or
  maintainer hardening rather than edits
- `harness-engineering` to tighten CLI, JSON, or end-to-end verification
  contracts
- `evidence-writing` only when cleaning maintainer prose after the technical
  content is correct

## Evidence discipline

Do not run this workflow from memory or from abstract preferences alone.
Ground the cycle in:

- canonical repo docs
- the changed code surface
- representative CLI evidence when the claim touches runtime behavior

The output should make clear which statements are verified facts, which are
inferences, and which are deferred follow-ups.

## Harness endpoints for this workflow

When hardening the workflow itself, treat these three endpoints as primary:

- `knowledge-integrity`
  - docs, routes, and source tables stay current and cross-linked
- `autonomy-capability`
  - agents can follow a bounded workflow with deterministic checks
- `architecture-invariants`
  - the guide keeps routing work toward canonical `reader` boundaries instead
    of smearing them together

## Workflow

### 1. Define the cycle scope

State:

- the target surface
- the gardening mode
- whether the cycle is audit-only or includes code or docs changes
- the workbench invariant you are protecting
- the representative assay family, experiment, or CLI surface if runtime
  verification is needed

Prefer one small slice over a repo-wide cleanup. If the target is broad, split
it by ownership boundary first.

### 2. Map current ownership

Trace the surface through the workbench layers described in
[ARCHITECTURE.md](../../ARCHITECTURE.md):

1. authored config
2. protocol semantics
3. compiled declaration and runtime execution
4. CLI, notebooks, records, and generated outputs

Use repository docs and implementation together. For runtime-facing mapping,
prefer machine-readable CLI evidence before making architectural claims:

```bash
uv run reader ls --root experiments --details --readiness --format json
uv run reader inspect <config|dir|index> --format json
uv run reader explain <config|dir|index> --format json
```

The goal is to name where meaning lives, where mechanics live, and where those
two are being confused.

### 3. Identify pressure and drift

Look for these failure modes:

- monolith pressure
  - one module or helper is collecting semantics, mechanics, and rendering
- assay lock-in
  - config or code assumes one assay family is the normal case
- stale semantics or docs
  - docs, routes, or invariants no longer match current behavior
- harness drift
  - CLI or JSON surfaces are brittle, inconsistent, or not fail-fast
- directory drift
  - ownership boundaries in code placement no longer match the architecture
- legacy creep
  - removed behavior or hidden fallback is trying to return through shims or
    ambiguous docs

Use the repo-local checklist at
[skills/reader-workbench-gardening/references/checklists.md](../../skills/reader-workbench-gardening/references/checklists.md)
to keep this pass concrete.

### 4. Choose the smallest reversible slice

Typical gardening moves are:

- move assay semantics out of generic runtime or CLI code
- split a growing family helper before it becomes the only place new assay work
  can land
- replace duplicated docs with canonical routes to existing architecture docs
- tighten validation or fail-fast behavior instead of carrying compatibility
  shims
- improve JSON or preflight surfaces so agents can inspect behavior without
  mutation

Avoid wide cleanup passes that mix unrelated layers. If the slice would touch
many ownership boundaries at once, it is probably too large.

When in doubt, prefer:

- a doc-route repair over a new overview document
- a fail-fast validation improvement over a compatibility shim
- a family-specific helper split over a generic abstraction that only moves the
  complexity
- one representative CLI contract improvement over a broad surface rewrite

### 5. Verify the slice

Use the smallest verification bundle that matches the risk. Start with the repo
change gate and then add representative runtime proof only where needed.

Docs or routing changes:

```bash
uv run python tools/audit_repo_skills.py
uv run python tools/check_docs.py
git diff --check
```

Code or CLI changes:

```bash
uv run ruff check .
uv run ruff format . --check
uv run pytest -q <targeted-tests>
git diff --check
```

Runtime, contract, or harness changes should also include a representative CLI
preflight path:

```bash
uv run reader ls --root experiments --details --readiness --format json
uv run reader inspect <config|dir|index> --format json
uv run reader validate <config|dir|index> --no-files --format json
uv run reader explain <config|dir|index> --format json
uv run reader run <config|dir|index> --dry-run --format json
```

When the gardening cycle changes end-to-end experiment behavior, add the
smallest repo marker that proves the surface:

```bash
uv run pytest -q -m repo_matrix
uv run pytest -q -m integration
uv run pytest -q -m active_experiments
```

Use only the smallest marker set that matches the risk. If plots, exports, or
notebooks changed, add the matching `plot --list`, `export --list`, `records`,
or notebook command for one representative experiment.

### 6. Close the cycle

Before finalizing:

1. review the diff
2. confirm no generated outputs were hand-edited
3. state skipped checks explicitly
4. route through [docs/repo-change-gate.md](../repo-change-gate.md) when
   tracked docs or code changed

If the task includes landing changes, do the normal branch, commit, and publish
steps after the gate passes. Commit and push are delivery steps, not the
identity of this workflow.

If the task expands into branch state, CI behavior, or remote publish steps,
continue into [Repo maintenance](../repo-maintenance.md) rather than trying to
hide that broader scope inside this guide.

## Deliverables

A good gardening cycle produces:

- the selected gardening mode
- a scoped statement of the invariant or boundary under review
- an evidence summary: canonical docs, code paths, and CLI probes used
- an ownership or drift summary
- the smallest reversible slice selected
- verification evidence
- residual risks and the next likely maintenance pass

## Related docs

- [Preflight, run, verify](./preflight_run_verify.md)
- [Automation and JSON](./automation.md)
- [Experiment bootstrap](./experiment_bootstrap.md)
- [Repo change gate](../repo-change-gate.md)
- [Repo maintenance](../repo-maintenance.md)
- [Architecture](../../ARCHITECTURE.md)
- [Design](../../DESIGN.md)
- [Quality](../../QUALITY.md)
- [Reliability](../../RELIABILITY.md)
