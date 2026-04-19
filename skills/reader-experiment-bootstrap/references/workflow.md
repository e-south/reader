# Workflow reference

Use this reference when the top-level skill needs a reminder of the concrete
`reader` commands and files involved in experiment bootstrapping.

## Discovery

```bash
uv run reader ls --root experiments --details --readiness
uv run reader ls --root experiments --details --readiness --format json
uv run reader inspect <config|dir|index>
uv run reader inspect <config|dir|index> --format json
uv run reader steps <config|dir|index>
uv run reader explain <config|dir|index>
uv run reader protocols <protocol-id>
```

Prefer JSON whenever another tool or agent will consume the output.

## Workspace creation

```bash
uv run reader init ./experiments/YYYY/YYYYMMDD_shortslug --protocol <protocol-id>
```

Use a copied nearest-neighbor config only when protocol defaults would lose
meaningful assay-specific behavior.

## Input intake

- Keep source filenames intact in `inputs/`.
- When Drive-backed intake is requested, use local `gws-account` commands rather
  than browser narration.
- Inspect workbook sheet names and channel labels before editing config.

## Metadata

- Preserve full plate coverage when the existing assay family expects it.
- Keep blanks explicit only when the assay semantics require blank subtraction
  or blank QC.
- Ask before resolving conflicting well assignments.

## Preflight

```bash
uv run reader validate <config|dir|index> --no-files
uv run reader validate <config|dir|index>
uv run reader run <config|dir|index> --dry-run --format json
uv run reader plot <config|dir|index> --list
uv run reader export <config|dir|index> --list
```

## Execution

```bash
uv run reader run <config|dir|index>
uv run reader plot <config|dir|index>
uv run reader export <config|dir|index>
uv run reader records <config|dir|index>
```

## Local experiment audit

```bash
uv run python tools/audit_local_experiments.py [--years <yyyy> [<yyyy> ...]] [--include-non-active]
```

This audit stages experiments into temporary copies so the original experiment
directories are not mutated during verification. By default it skips non-active
lifecycles and reports them separately.
