---
doc_id: reader-automation-json
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-08-01
summary: Versioned success and failure envelopes for Reader automation.
---

# Automation and JSON

Use JSON output when another tool needs stable discovery, inspection, or
preflight data from `reader`.

Every JSON response uses the `reader.cli/v1` envelope:

```json
{
  "schema": "reader.cli/v1",
  "ok": true,
  "command": "inspect",
  "data": {},
  "error": null,
  "meta": {"projection": "full", "truncated": false, "continuation": null}
}
```

Read command-specific fields from `data` only when `ok` is `true`. A failed
command exits nonzero, writes exactly one JSON document to stdout, leaves
stderr empty, and sets `data` to `null`. Its `error` object always contains
`code`, `field`, `reason`, `remediation`, and `retryable`. This includes
argument-parsing, invalid-parameter, and Reader runtime failures.

## Bounded responses

Large collections return at most 25 entries per JSON page by default. This
applies to `ls`, `plugins`, and `records`; it does not change their table
output. Set `--limit` from 1 through 100, then pass the opaque token from
`meta.continuation` back through `--continuation` until `meta.truncated` is
`false`:

```bash
uv run reader ls --root experiments --details --limit 10 --format json
uv run reader ls --root experiments --details --limit 10 \
  --continuation <token> --format json
uv run reader plugins --category transform --limit 10 --format json
uv run reader records <config|dir|index> --limit 10 --format json
```

The summary describes the complete filtered collection while the collection
array contains the current page. A continuation is bound to its command and
filters; changing the root, readiness mode, plugin filters, experiment, or
record-history mode requires a new first page. Record continuations are also
bound to the current config, provenance epoch, and visible record revisions;
if any of those change, restart from the first page.

Large single-object descriptions use semantic projections instead of JSON
paths. A selected projection is reported as `section:<name>` in
`meta.projection`.

## Experiment list

```bash
uv run reader ls --root experiments --details --readiness --format json
```

Use this as the machine-readable experiment list with readiness data. It includes
`catalog`, `selection`, `summary`, and `experiments`.

## Protocol discovery

```bash
uv run reader protocols <protocol-id> --format json
uv run reader protocols <protocol-id> --section authoring --format json
uv run reader protocols <protocol-id> --section semantics --format json
uv run reader protocols <protocol-id> --section compiled --format json
uv run reader plugins --protocol <protocol-id> --category <category> --format json
```

Use `protocols` for the public assay definition and compiled defaults. Use
`--section identity|authoring|semantics|defaults|compiled` when only one part is
needed. Use `plugins` only when you need registry-level inspection for one
protocol.

## Single experiment inspection

```bash
uv run reader inspect <config|dir|index> --format json
uv run reader inspect <config|dir|index> --section readiness --format json
uv run reader inspect <config|dir|index> --section plan --format json
uv run reader config <config|dir|index> --format json
uv run reader steps <config|dir|index> --format json
uv run reader explain <config|dir|index> --format json
```

These commands use shared `authoring`, `semantics`, and `implementation`
sections so one experiment can be inspected without scraping table output.
`inspect --section` accepts `identity`, `authoring`, `semantics`, `plan`,
`compiled`, `inputs`, `generated`, or `readiness`. The projection always keeps
experiment identity beside the selected section.

## Preflight and dry-run

```bash
uv run reader validate <config|dir|index> --no-files --format json
uv run reader validate <config|dir|index> --format json
uv run reader run <config|dir|index> --dry-run --format json
uv run reader plot <config|dir|index> --list --format json
uv run reader plot <config|dir|index> --dry-run --format json
uv run reader export <config|dir|index> --list --format json
uv run reader export <config|dir|index> --dry-run --format json
```

Use `validate --no-files` for schema and wiring only, `validate` when input
files matter, and `run --dry-run` to inspect the execution slice without
mutation. Plot and export dry-runs provide the same no-write JSON planning
surface and validate their log level and selected specs before returning.

## Record verification

```bash
uv run reader verify <config|dir|index> --format json
```

A verified catalog returns the `reader.verify/v1` report inside `data`.
Corruption, drift, missing evidence, and missing catalogs use the standard
failure envelope and a nonzero exit status.

## Records

```bash
uv run reader records <config|dir|index> --format json
```

Use `records` to inspect the manifest path, record summary counts, persisted
descriptions, and optional revision history for one experiment. New plot
records map every path to the matching protocol figure summary or explicit
producer metadata; export records retain the producing plugin's operational
bundle description. Notebook deliverables read those persisted descriptions
instead of deriving meaning from filenames.

## When not to use JSON

Use the default table output when the task is interactive and the current
question is easier to answer from the terminal. Switch to JSON when another
tool needs deterministic fields.

## Related docs

- [Preflight, run, verify](./preflight_run_verify.md)
- [Common tasks](./common_routes.md)
- [CLI reference](../core/cli.md)
