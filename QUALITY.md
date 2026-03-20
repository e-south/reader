# Quality

`reader` quality is not just “tests pass.” The quality bar is that users and agents can discover, validate, dry-run, execute, and verify experiments through explicit contracts with low ambiguity and low retry cost.

This document defines that bar.

## Quality Model

The workbench is healthy when these questions have fast, deterministic answers.

- Can I discover what experiments exist?
- Can I inspect what one experiment is configured to do?
- Can I validate the config without running the workflow?
- Can I dry-run the planned execution slice?
- Can I execute and then verify records, plots, and exports with provenance?
- Can I extend the system without widening the public config into a junk drawer?

## Harness Contract

- Objective:
  Keep `reader` clear for experiment authors and mechanically legible for maintainers and agents.
- Scope boundary:
  In scope are the CLI, docs, config surface, protocol catalog, runtime planning, validation, and records.
  Out of scope are ad hoc manual edits to generated outputs.
- Acceptance checks:
  Discovery, inspect, validate, explain, dry-run, plot/export listing, and record inspection must remain available and deterministic.
- Stop conditions:
  Halt when a proposed change requires reviving legacy config shims, plugin-shaped public config, or hidden mutation paths.
- Escalation criteria:
  Escalate when assay semantics cannot be expressed without widening the public surface or violating architecture boundaries.

## In-Scope Endpoints

The top-level quality program for `reader` currently centers on three harness endpoints.

### `knowledge-integrity`

- Docs must be current, cross-linked, and aligned with the actual CLI and config surface.
- README and docs index must route readers to setup, preflight/run/verify, automation, and maintainer paths without duplication.

### `autonomy-capability`

- Core commands must support deterministic machine-readable inspection where appropriate.
- Agents should not need to scrape Rich tables to understand the workbench.

### `architecture-invariants`

- Public config stays `reader/v7`.
- Protocols stay the semantic owner.
- Plugins stay mechanical adapters.
- Generated outputs remain generated.

## Fast Feedback Loop

The preferred quality loop is:

1. `reader ls`
2. `reader inspect`
3. `reader validate --no-files`
4. `reader explain`
5. `reader run --dry-run`
6. targeted execution
7. `reader records`, `reader plot --list`, `reader export --list`

This follows the same harness lesson emphasized in OpenAI’s harness-engineering article: speed and explicit feedback loops matter because slow or ambiguous verification causes retries and drift.

Use [docs/guides/preflight_run_verify.md](./docs/guides/preflight_run_verify.md) for the task-oriented version of this loop.

## Required Evidence For Changes

The minimum verification bundle for workbench-facing changes should include:

- `uv run python tools/check_docs.py`
- targeted CLI command coverage for the changed surface
- targeted tests for changed behavior
- `uv run ruff check .`
- `uv run ruff format . --check`
- `uv run python -m compileall src/reader`
- `git diff --check`

For documentation-only changes, at least verify:

- `uv run python tools/check_docs.py`
- linked routes are still valid
- command examples match current CLI behavior
- changed docs stay aligned with `reader/v7`

## Quality Gates

### Config and schema quality

- `yaml.safe_load`
- strict `reader/v7` schema check
- removed legacy keys rejected explicitly
- pydantic models forbid extra fields on plugin configs

### Runtime quality

- typed input and output ports
- explicit dataframe contracts
- validation of reads/writes compatibility
- record manifests with content digests and provenance

### UX quality

- clear routing from overview pages to workflow guides and detailed reference
- machine-readable JSON surfaces for core discovery and preflight paths
- explicit empty-state and failure-state behavior

## Failure Taxonomy

These are the failure classes that quality work should continue to reduce.

- semantic drift
  Protocol meaning lives partly in one place and execution truth in another.
- table-only surface
  Important inspection or preflight paths exist only in table output.
- silent fallback
  Broken configs or empty selections appear as success without explicit signal.
- junk-drawer config growth
  New capabilities are added by widening generic bags instead of improving protocol semantics.
- provenance opacity
  Outputs exist on disk but their producing steps or inputs are hard to recover.

## Current Open Quality Debt

The main unresolved quality debt is still semantic, not documentation. Protocol windows, controls, metrics, and ranking are not yet compiled from one executable typed analysis program. Until that changes, there is still some duplicate truth between protocol metadata and compiler behavior.

## Definition Of Done

A workbench change is done when:

- the public surface is still cleaner or tighter than before
- the verification path is explicit
- generated outputs were not hand-edited
- docs and CLI agree
- targeted tests and checks pass or any gap is explicitly documented

## Related Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [DESIGN.md](./DESIGN.md)
- [RELIABILITY.md](./RELIABILITY.md)
- [SECURITY.md](./SECURITY.md)
