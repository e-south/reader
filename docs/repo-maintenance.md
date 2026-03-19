# Repo Maintenance

This document is the maintainer-facing route map for repo-wide changes, publish flow, and ongoing workbench hygiene.

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

## CI Topology

`reader` uses two GitHub Actions workflows with explicit intent:

- `CI` in [.github/workflows/ci.yaml](../.github/workflows/ci.yaml): fast maintainer feedback on pull requests and pushes. It runs lockfile drift checks, lint, format, compile, build, the default pytest lane with coverage, and a separate repo-contract job for config and cytometry integration checks.
- `Integration` in [.github/workflows/integration.yaml](../.github/workflows/integration.yaml): slow repo-operator validation on `main`, nightly schedule, and manual dispatch. It runs `uv run pytest -q -m integration --durations=25` and uploads the experiment readiness inventory as an artifact.

Local command contract:

- `uv run pytest -q`: fast default lane, excludes `integration`
- `uv run pytest -q -m smoke`: representative real-experiment smoke tests
- `uv run pytest -q -m integration`: repo-wide config and end-to-end experiment matrix
