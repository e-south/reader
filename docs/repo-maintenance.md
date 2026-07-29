---
doc_id: reader-repo-maintenance
surface: maintainer-runbook
owner: reader-maintainers
last_verified: 2026-07-28
summary: Broader Reader branch, CI, repository health, and delivery workflow beyond the minimum change gate.
---

# Repo Maintenance

This document is the maintainer guide for repo-wide changes, publish flow, and ongoing repo hygiene.

## Use This When

- the change crosses multiple package boundaries
- branch or publish state matters
- CI or verification policy needs to change
- docs, CLI, and runtime behavior need to stay in sync across the repo

For the smaller tracked-change workflow, start with [repo-change-gate.md](./repo-change-gate.md).

## Reference Points

- Repo entry point:
  [README.md](../README.md)
- Authoritative docs map:
  [docs/README.md](./README.md)
- System structure:
  [ARCHITECTURE.md](../ARCHITECTURE.md)
- Product and information-design rules:
  [DESIGN.md](../DESIGN.md)
- Quality program and evidence expectations:
  [QUALITY.md](../QUALITY.md)
- Operational reliability loop:
  [RELIABILITY.md](../RELIABILITY.md)
- Trust boundaries and safe defaults:
  [SECURITY.md](../SECURITY.md)

## Repo-Wide Expectations

- Keep the workbench discoverable from the CLI before requiring people to read source.
- Prefer explicit registries and typed contracts over implicit discovery.
- Keep docs aligned with the actual runtime and CLI behavior.
- Keep protocol semantics tighter than plugin mechanics.
- Favor small, reviewable changes over broad rewrites unless the broader cut is clearly justified.

## Verification Strategy

Choose the smallest verification set that still exercises the risk:

- docs and routing
- CLI discovery and preflight
- runtime planning and execution
- plugin or contract boundary changes
- repo-wide smoke and lint checks

The quality bar for those checks is defined in [QUALITY.md](../QUALITY.md).

For docs and routing changes, start with:

```bash
uv run reader maintain docs
git diff --check
```

## Automation Topology

Reader keeps continuous verification and publication distinct:

- `Checks` in [.github/workflows/checks.yaml](../.github/workflows/checks.yaml)
  runs on pull requests and main pushes. Package verification, supported Typer
  bounds, and the portable test suite run in parallel; the dependency audit
  covers both runtime and notebook execution surfaces, and coverage upload
  remains separately permissioned. The final `Checks` job is the only required
  branch-protection context.
- `Release` in [.github/workflows/release.yaml](../.github/workflows/release.yaml)
  runs only for a published GitHub release whose `v<version>` tag exactly
  matches `pyproject.toml`. An unprivileged job builds the distributions, then
  a `pypi` environment job publishes them through OIDC. PyPI must trust
  `e-south/reader`, workflow `release.yaml`, and environment `pypi` before the
  first release.

Local commands:

- `uv run ruff check .`: repo-wide lint
- `uv run ruff format . --check`: formatting check
- `uv run reader maintain docs`: docs links and routing integrity
- `uv run pytest -q`: portable default test run
- `uv run pytest -q -m integration`: portable cross-surface integration tests
- `uv run pytest -q -m active_experiments`: optional local data-backed checks;
  requires ignored experiment inputs and is not part of public CI
- `git diff --check`: whitespace and merge-marker hygiene
