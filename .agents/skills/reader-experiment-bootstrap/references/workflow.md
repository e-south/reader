# Workflow reference

Use this reference when the top-level skill needs a reminder of the concrete
`reader` commands and files involved in experiment bootstrapping.

## Discovery

Start with the Data Operations Plan overview and load only the reference needed
for the current decision:

- [Data classes](../../../../docs/guides/data_operations_plan/data_classes.md)
  for the class/protocol decision.
- [Metadata minimums](../../../../docs/guides/data_operations_plan/metadata_minimums.md)
  when building or reviewing sample maps and config metadata.
- [Transfer and verification](../../../../docs/guides/data_operations_plan/transfer_and_verification.md)
  when staging inputs and proving outputs.

```bash
uv run reader dop classes --format json
uv run reader dop ready-specs --format json
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
- Use the Data Operations Plan stop conditions when well identity, treatment
  meaning, controls, or channel semantics are ambiguous.

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
uv run reader audit experiments [--years <yyyy>] [--include-non-active]
```

This audit stages experiments into temporary copies so the original experiment
directories are not mutated during verification. By default it skips non-active
lifecycles and reports them separately.
