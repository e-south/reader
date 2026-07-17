---
doc_id: reader-repo-change-gate
surface: maintainer-runbook
owner: reader-maintainers
last_verified: 2026-07-17
summary: Minimum review and verification gate for tracked Reader code, docs, tests, and CLI changes.
---

# Repo Change Gate

Use this as the minimum maintainer gate before landing tracked changes in `reader`.

## Scope

This gate is for ordinary repo-local changes to code, docs, tests, or CLI behavior. It is the shortest path for checking that a change is reviewable and does not break the expected workbench behavior.

For broader repo behavior, publish flow, or CI topology, continue to [repo-maintenance.md](./repo-maintenance.md).

## Minimum Gate

Before finalizing a change:

1. Review the diff.
2. Confirm you did not hand-edit `experiments/**/outputs/`.
3. Run the smallest verification set that matches the change:
   - docs-only: run `uv run python tools/check_docs.py` and `git diff --check`
   - CLI/code: targeted tests plus `uv run ruff check .`, `uv run ruff format . --check`, and `git diff --check`
   - runtime or contract changes: targeted tests, `uv run ruff check .`, `uv run ruff format . --check`, a representative CLI preflight command, and `git diff --check`
4. State any skipped verification explicitly.

## Non-Negotiable Invariants

- Public config stays `reader/v8`.
- Removed config keys do not return through aliases or shims.
- Protocols own assay-facing semantics.
- Plugins remain mechanical adapters, not the public UX.
- Generated outputs remain generated.

## Canonical Verification Commands

```bash
uv run python tools/check_docs.py
uv run ruff check .
uv run ruff format . --check
uv run pytest -q
uv run pytest -q -m smoke
uv run pytest -q -m repo_matrix
uv run pytest -q -m active_experiments
uv run pytest -q -m integration
git diff --check
```

`uv run pytest -q` is the fast default test run: it excludes only the full data-backed active-experiment run while still covering ordinary integration checks and the repo-wide config sweep. Use `uv run pytest -q -m repo_matrix` when the change mainly touches repo config invariants, `uv run pytest -q -m active_experiments` for the full active-experiment end-to-end run, and `uv run pytest -q -m integration` when you intentionally want the full integration set. Use the smallest subset that matches the risk of the change and explain any omission.

## Related Docs

- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [DESIGN.md](../DESIGN.md)
- [QUALITY.md](../QUALITY.md)
- [RELIABILITY.md](../RELIABILITY.md)
- [SECURITY.md](../SECURITY.md)
