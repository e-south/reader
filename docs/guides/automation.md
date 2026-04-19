# Automation and JSON

Use JSON output when another tool needs stable discovery, inspection, or
preflight data from `reader`.

## Experiment list

```bash
uv run reader ls --root experiments --details --readiness --format json
```

Use this as the machine-readable experiment list with readiness data. It includes
`catalog`, `selection`, `summary`, and `experiments`.

## Protocol discovery

```bash
uv run reader protocols <protocol-id> --format json
uv run reader plugins --protocol <protocol-id> --category <category> --format json
```

Use `protocols` for the public assay definition and compiled defaults. Use
`plugins` only when you need registry-level inspection for one protocol.

## Single experiment inspection

```bash
uv run reader inspect <config|dir|index> --format json
uv run reader config <config|dir|index> --format json
uv run reader steps <config|dir|index> --format json
uv run reader explain <config|dir|index> --format json
```

These commands use shared `authoring`, `semantics`, and `implementation`
sections so one experiment can be inspected without scraping table output.

## Preflight and dry-run

```bash
uv run reader validate <config|dir|index> --no-files --format json
uv run reader validate <config|dir|index> --format json
uv run reader run <config|dir|index> --dry-run --format json
uv run reader plot <config|dir|index> --list --format json
uv run reader export <config|dir|index> --list --format json
```

Use `validate --no-files` for schema and wiring only, `validate` when input
files matter, and `run --dry-run` to inspect the execution slice without
mutation.

## Records

```bash
uv run reader records <config|dir|index> --format json
```

Use `records` to inspect the manifest path, record summary counts, and optional
revision history for one experiment.

## When not to use JSON

Use the default table output when the task is interactive and the current
question is easier to answer from the terminal. Switch to JSON when another
tool needs deterministic fields.

## Related docs

- [Preflight, run, verify](./preflight_run_verify.md)
- [Common tasks](./common_routes.md)
- [CLI reference](../core/cli.md)
