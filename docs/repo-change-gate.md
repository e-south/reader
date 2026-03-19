# Repo Change Gate

Use this as the minimum maintainer gate before landing tracked changes in `reader`.

## Scope

This gate is for ordinary repo-local changes to code, docs, tests, or CLI behavior. It is the shortest path for checking that a change is reviewable and does not violate the workbench contract.

For deeper repo surfaces, publish flow, or CI topology, continue to [repo-maintenance.md](./repo-maintenance.md).

## Minimum Gate

Before finalizing a change:

1. Review the diff.
2. Confirm you did not hand-edit `experiments/**/outputs/`.
3. Run the smallest verification bundle that matches the change:
   - docs-only: validate referenced routes and run `git diff --check`
   - CLI/code: targeted tests plus `uv run ruff check .`
   - runtime/contract changes: targeted tests, lint, and a representative CLI preflight path
4. State any skipped verification explicitly.

## Non-Negotiable Invariants

- Public config stays `reader/v7`.
- Removed legacy keys do not come back through compatibility shims.
- Protocols own assay-facing semantics.
- Plugins remain mechanical adapters, not the public UX.
- Generated outputs remain generated.

## Canonical Verification Commands

```bash
uv run ruff check .
uv run ruff format . --check
uv run pytest -q
uv run pytest -q -m smoke
uv run pytest -q -m repo_matrix
uv run pytest -q -m fleet
uv run pytest -q -m integration
git diff --check
```

`uv run pytest -q` is the fast default lane: it excludes only the full data-backed `fleet` matrix while still running ordinary integration coverage and the repo-wide config sweep. Use `uv run pytest -q -m repo_matrix` when the change mainly touches repo config invariants, `uv run pytest -q -m fleet` for the full active-experiment end-to-end matrix, and `uv run pytest -q -m integration` when you intentionally want the full integration surface. Use the smallest subset that matches the risk of the change and explain any omission.

## Related Docs

- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [DESIGN.md](../DESIGN.md)
- [QUALITY.md](../QUALITY.md)
- [RELIABILITY.md](../RELIABILITY.md)
- [SECURITY.md](../SECURITY.md)
