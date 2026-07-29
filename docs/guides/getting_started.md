---
doc_id: reader-getting-started
surface: tutorial
owner: reader-maintainers
last_verified: 2026-07-28
summary: Minimal setup and first-inspection path for a new Reader user.
---

# Getting started

Reader uses `uv` for installation and command execution. Start with the guided
command tour, then discover a protocol or inspect an existing experiment. None
of these steps changes a real experiment.

## Install the environment

```bash
uv sync --locked
```

This creates or updates `.venv` with Reader and its runtime dependencies.

## Run the guided demo

```bash
uv run reader demo
```

The demo prints a 14-command tour. It does not execute those commands, use
example data, or write files.

## Discover protocols and experiments

```bash
uv run reader protocols
uv run reader ls --root experiments
uv run reader ls --root experiments --details --readiness
```

`reader protocols` lists the assay types Reader knows how to scaffold. Start
with `reader ls` for existing experiments; add `--details --readiness` when you
need protocol, output, and preflight state in the same view.

## Scaffold a new experiment

```bash
uv run reader init ./experiments/my_experiment --protocol plate_reader/single_reporter_screen
uv run reader validate ./experiments/my_experiment/config.yaml --no-files
```

The starter contains only protocol-owned defaults and declared input
resources; Reader creates `outputs/` only when a real run writes results. Edit
the metadata and protocol choices before adding instrument files to `inputs/`.

## Inspect one experiment before execution

```bash
uv run reader inspect experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
uv run reader validate experiments/2025/20250614_sensor_panel_M9_glu/config.yaml --no-files
uv run reader explain experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
```

Use `inspect` for the bound experiment summary, `validate` for preflight checks, and `explain` for the compiled runtime plan.

## Maintainer checks

Contributors should use the [repo change gate](../repo-change-gate.md) for the
current test, lint, formatting, documentation, and build commands. New users do
not need to run the full maintainer suite before trying Reader.

## Continue from here

- [Common tasks](./common_routes.md)
- [Preflight, run, verify](./preflight_run_verify.md)
- [Automation and JSON](./automation.md)
- [CLI reference](../core/cli.md)
- [Configuring `reader/v8`](../core/pipeline.md)
- [Notebooks](./notebooks.md)
