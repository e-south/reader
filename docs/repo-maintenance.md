---
doc_id: reader-repo-maintenance
surface: maintainer-runbook
owner: reader-maintainers
last_verified: 2026-07-10
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
uv run python tools/check_docs.py
git diff --check
```

## CI Topology

`reader` uses two GitHub Actions workflows:

- `CI` in [.github/workflows/ci.yaml](../.github/workflows/ci.yaml): pull-request and push feedback. It runs docs integrity, lockfile drift checks, lint, format, compile, build, and the default test run with coverage. The default run is `uv run pytest -q`, which excludes only the active-experiment run.
- `Integration` in [.github/workflows/integration.yaml](../.github/workflows/integration.yaml): slower main-branch, nightly, and manual validation. It runs `pytest -m integration` with `--durations=25` and uploads the experiment readiness summary as an artifact.

Local commands:

- `uv run ruff check .`: repo-wide lint
- `uv run ruff format . --check`: formatting check
- `uv run python tools/check_docs.py`: docs links and routing integrity
- `uv run pytest -q`: fast default test run, excludes only the active-experiment run
- `uv run pytest -q -m repo_matrix`: repo-wide config and metadata sweeps
- `uv run pytest -q -m smoke`: representative real-experiment smoke tests
- `uv run pytest -q -m active_experiments`: full active-experiment end-to-end run
- `uv run pytest -q -m integration`: full integration set, including `repo_matrix` and `active_experiments`
- `git diff --check`: whitespace and merge-marker hygiene
