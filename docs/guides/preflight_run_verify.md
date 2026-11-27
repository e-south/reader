---
doc_id: reader-preflight-run-verify
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-11-27
summary: Canonical non-mutating preflight, execution, and provenance-verification loop for one experiment.
---

# Preflight, run, verify

Use a configured experiment with its input files in place. The commands below
use the [sample plate](./demo.md). Set `exp` to your own experiment directory
to run them on your files. In a source checkout, prefix `reader` with `uv run`.

## 1. Discover the experiment

```bash
exp=./experiments/first_plate
reader ls --root experiments --details --readiness
```

Use `reader ls` to confirm the experiment exists, which protocol it uses, and
whether it is blocked, draft, runnable, or already has records.

## 2. Inspect the plan before execution

```bash
reader inspect "$exp"
reader steps "$exp"
reader explain "$exp"
```

Use `inspect` for the full experiment summary, `steps` for the pipeline chain,
and `explain` for the compiled runtime plan.

## 3. Run the cheapest preflight check that answers the question

```bash
reader validate "$exp" --no-files
reader validate "$exp"
reader run "$exp" --dry-run
reader plot "$exp" --list
reader export "$exp" --list
```

- Use `validate --no-files` for schema and wiring only.
- Use `validate` when file availability matters.
- Use `run --dry-run` to confirm the execution slice without mutation.
- Use `plot --list` and `export --list` to confirm resolved outputs before
  generating them.

## 4. Execute only the slice you need

```bash
reader run "$exp"
reader plot "$exp"
reader export "$exp"
reader notebook "$exp" --mode none
reader notebook "$exp" --mode run --headless
```

`run` materializes records. `plot` and `export` materialize their output
surfaces after the experiment is ready. `notebook --mode none` scaffolds a
review notebook without launching Marimo, and `--mode run --headless` prints a
loopback URL for agent/browser review.

## 5. Verify outputs and provenance

```bash
reader records "$exp"
reader verify "$exp"
reader inspect "$exp"
```

`records` is the catalog view. `verify` checks the current config, source files,
upstream record revisions, generated files, recorded digests, and invocation
lifecycle. A zero exit status is the machine-checkable proof that current
schema-v6 records still match those inputs and outputs and that every recorded
attempt has one unambiguous terminal result.

For a human review, also inspect:

- `outputs/manifests/records.json`
- the active `outputs/manifests/invocations/<epoch>.jsonl`
- generated plot files
- generated export files

The record catalog proves what was produced; the invocation ledger records each
attempt and terminal result. Neither requires guessing from filesystem state or
using `JOURNAL.md` as a machine log.

## Recovery loop

When something fails:

1. Fix the config or code.
2. Rerun the cheapest preflight command that checks the changed surface.
3. Rerun only the affected execution slice.
4. Verify records and outputs again.

## Related docs

- [Getting started](./getting_started.md)
- [Common tasks](./common_routes.md)
- [Automation and JSON](./automation.md)
- [CLI reference](../core/cli.md)
