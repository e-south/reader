# Repo Maintenance

This document is the maintainer guide for repo-wide changes, publish flow, and ongoing workbench hygiene.

## Use This When

- the change crosses multiple package boundaries
- branch or publish state matters
- CI or verification policy needs to change
- docs, CLI, and runtime contracts need to be kept in sync across the repo

For the smaller tracked-change workflow, start with [repo-change-gate.md](./repo-change-gate.md).

## Maintenance Surfaces

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
- Keep docs aligned with the actual runtime and CLI surface.
- Keep protocol semantics tighter than plugin mechanics.
- Favor small, reviewable changes over broad rewrites unless the rooted cut is clear.

## Verification Strategy

Choose the cheapest verification bundle that still exercises the risk:

- docs and routing
- CLI discovery and preflight
- runtime planning and execution
- plugin or contract boundary changes
- repo-wide smoke and lint checks

The quality bar for those bundles is defined in [QUALITY.md](../QUALITY.md).

For docs and routing changes, start with:

```bash
uv run python tools/check_docs.py
git diff --check
```

## CI Topology

`reader` uses two GitHub Actions workflows:

- `CI` in [.github/workflows/ci.yaml](../.github/workflows/ci.yaml): pull-request and push feedback. It runs docs integrity, lockfile drift checks, lint, format, compile, build, and the default test lane with coverage. The default lane is `uv run pytest -q`, which excludes only `fleet`.
- `Integration` in [.github/workflows/integration.yaml](../.github/workflows/integration.yaml): slower main-branch, nightly, and manual validation. It runs `pytest -m integration` with `--durations=25` and uploads the experiment readiness inventory as an artifact.

Local command contract:

- `uv run python tools/check_docs.py`: docs links and routing integrity
- `uv run pytest -q`: fast default lane, excludes only `fleet`
- `uv run pytest -q -m repo_matrix`: repo-wide config and metadata sweeps
- `uv run pytest -q -m smoke`: representative real-experiment smoke tests
- `uv run pytest -q -m fleet`: full active-experiment end-to-end matrix
- `uv run pytest -q -m integration`: full integration surface, including `repo_matrix` and `fleet`
