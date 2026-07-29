---
doc_id: reader-common-routes
surface: task-router
owner: reader-maintainers
last_verified: 2026-07-28
summary: Short command routes for common Reader discovery, preflight, execution, output, and recovery tasks.
---

# Common tasks

Use this page when you know what you want to do and need the shortest command path.

## List experiments

```bash
uv run reader ls --root experiments
uv run reader ls --root experiments --details
uv run reader ls --root experiments --details --readiness
```

The first command lists discovered experiments. `--details` adds selected
pipeline, plot, and export summaries. `--readiness` adds the current preflight
state, including `config_error`, `template`, `draft`, `dependency_blocked`,
`blocked`, `runnable`, `uncataloged_outputs_present`, and `records_ready`.
`catalog_ready` means a valid schema-v5 catalog exists but its recorded config
or Reader build differs from the current environment. `records_ready` requires
every current record to pass source, config, dependency, and generated-file
verification; an empty or invalid catalog is not sufficient.

## Inspect one experiment

```bash
uv run reader inspect <config|dir|index>
uv run reader steps <config|dir|index>
uv run reader explain <config|dir|index>
```

Use `inspect` for the full summary, `steps` for the pipeline chain only, and `explain` for the compiled runtime plan.

## Discover protocols and scaffold a new experiment

```bash
uv run reader protocols
uv run reader protocols <protocol-id>
uv run reader protocols <protocol-id> --example-config
uv run reader init ./experiments/<new-experiment> --protocol <protocol-id>
```

Use the protocol commands to see the public assay definition before you scaffold a new experiment.

## Validate before execution

```bash
uv run reader validate <config|dir|index>
uv run reader validate <config|dir|index> --no-files
uv run reader run <config|dir|index> --dry-run
```

`validate --no-files` is the cheapest preflight path when you only need schema and wiring checks. `run --dry-run` shows the execution slice without mutating outputs.

## Materialize outputs

```bash
uv run reader run <config|dir|index>
uv run reader plot <config|dir|index> --list
uv run reader plot <config|dir|index>
uv run reader export <config|dir|index> --list
uv run reader export <config|dir|index>
uv run reader records <config|dir|index>
uv run reader verify <config|dir|index>
```

Run the pipeline first, then list or render plots and exports as needed.
`reader records` inspects the catalog; `reader verify` proves the current
evidence still matches its sources and files.

## Use JSON for automation

```bash
uv run reader ls --root experiments --details --readiness --format json
uv run reader inspect <config|dir|index> --format json
uv run reader validate <config|dir|index> --no-files --format json
uv run reader run <config|dir|index> --dry-run --format json
```

Use JSON output when another tool needs stable machine-readable discovery, inspection, or preflight data.

## Continue to the full reference

- [Preflight, run, verify](./preflight_run_verify.md)
- [Automation and JSON](./automation.md)
- [CLI reference](../core/cli.md)
- [Configuring `reader/v8`](../core/pipeline.md)
- [End-to-end demo](./demo.md)
