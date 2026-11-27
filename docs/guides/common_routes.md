---
doc_id: reader-common-routes
surface: task-router
owner: reader-maintainers
last_verified: 2026-11-27
summary: Short command routes for common Reader discovery, preflight, execution, output, and recovery tasks.
---

# Common tasks

These commands use the installed CLI. In a source checkout, use `uv run reader`.
For commands that take an experiment, set its path once. This example uses
the [sample plate](./demo.md).

```bash
exp=./experiments/first_plate
```

## List experiments

```bash
reader ls --root experiments
reader ls --root experiments --details
reader ls --root experiments --details --readiness
```

The first command lists discovered experiments. `--details` adds selected
pipeline, plot, and export summaries. `--readiness` adds the current preflight
state, including `config_error`, `template`, `draft`, `dependency_blocked`,
`blocked`, `runnable`, `uncataloged_outputs_present`, and `records_ready`.
`catalog_ready` means a valid catalog-schema-v4 envelope with record-schema-v6
payloads exists but its recorded config or Reader build differs from the
current environment. `records_ready` requires
every current record to pass source, config, dependency, and generated-file
verification; an empty or invalid catalog is not sufficient.

## Inspect one experiment

```bash
reader inspect "$exp"
reader steps "$exp"
reader explain "$exp"
```

Use `inspect` for the full summary, `steps` for the pipeline chain only, and `explain` for the compiled runtime plan.

## Discover protocols and scaffold a new experiment

```bash
reader protocols
protocol=plate_reader/single_reporter_screen
reader protocols "$protocol"
reader protocols "$protocol" --example-config
reader init ./experiments/another_plate --protocol "$protocol"
```

Use the protocol commands to see the public assay definition before you scaffold a new experiment.

## Validate before execution

```bash
reader validate "$exp"
reader validate "$exp" --no-files
reader run "$exp" --dry-run
```

`validate --no-files` is the cheapest preflight path when you only need schema and wiring checks. `run --dry-run` shows the execution slice without mutating outputs.

## Materialize outputs

```bash
reader run "$exp"
reader plot "$exp" --list
reader plot "$exp"
reader export "$exp" --list
reader records "$exp"
reader verify "$exp"
```

The sample plate defines two plots. If your config also lists exports, run
`reader export "$exp"` to write them. `reader records` lists the catalog
entries, and `reader verify` checks their inputs and files.

## Use JSON for automation

```bash
reader ls --root experiments --details --readiness --format json
reader inspect "$exp" --format json
reader validate "$exp" --no-files --format json
reader run "$exp" --dry-run --format json
```

Use JSON output when another tool needs stable machine-readable discovery, inspection, or preflight data.

## Continue to the full reference

- [Preflight, run, verify](./preflight_run_verify.md)
- [Automation and JSON](./automation.md)
- [CLI reference](../core/cli.md)
- [Configuring `reader/v8`](../core/pipeline.md)
- [Run a sample plate](./demo.md)
