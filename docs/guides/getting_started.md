# Getting started

`reader` uses `uv` for environment management and command execution. Install the project, verify the checkout, then inspect an experiment before running anything.

## Install the environment

```bash
uv sync --locked --group dev --group notebooks
```

This creates or updates `.venv` with the package, test tools, and notebook dependencies.

## Check the checkout

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format . --check
```

`uv run pytest -q` is the default test lane. It excludes only the full data-backed `fleet` matrix. The Ruff commands check lint and formatting.

## Inspect the experiment inventory

```bash
uv run reader ls --root experiments
uv run reader ls --root experiments --details --readiness
```

Start with `reader ls` to see the experiment catalog. Add `--details --readiness` when you need protocol, output, and readiness state in the same view.

## Inspect one experiment before execution

```bash
uv run reader inspect experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
uv run reader validate experiments/2025/20250614_sensor_panel_M9_glu/config.yaml --no-files
uv run reader explain experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
```

Use `inspect` for the bound experiment summary, `validate` for preflight checks, and `explain` for the compiled runtime plan.

## Continue from here

- [Common tasks](./common_routes.md)
- [Preflight, run, verify](./preflight_run_verify.md)
- [Automation and JSON](./automation.md)
- [CLI reference](../core/cli.md)
- [Configuring `reader/v7`](../core/pipeline.md)
- [Notebooks](./notebooks.md)
