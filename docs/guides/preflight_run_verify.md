---
doc_id: reader-preflight-run-verify
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-07-10
summary: Canonical non-mutating preflight, execution, and provenance-verification loop for one experiment.
---

# Preflight, run, verify

Use this path when you need the shortest reliable route from inspection to
execution for one experiment.

## 1. Discover the experiment

```bash
uv run reader ls --root experiments --details --readiness
```

Use `reader ls` to confirm the experiment exists, which protocol it uses, and
whether it is blocked, draft, runnable, or already has records.

## 2. Inspect the plan before execution

```bash
uv run reader inspect <config|dir|index>
uv run reader steps <config|dir|index>
uv run reader explain <config|dir|index>
```

Use `inspect` for the full experiment summary, `steps` for the pipeline chain,
and `explain` for the compiled runtime plan.

## 3. Run the cheapest preflight check that answers the question

```bash
uv run reader validate <config|dir|index> --no-files
uv run reader validate <config|dir|index>
uv run reader run <config|dir|index> --dry-run
uv run reader plot <config|dir|index> --list
uv run reader export <config|dir|index> --list
```

- Use `validate --no-files` for schema and wiring only.
- Use `validate` when file availability matters.
- Use `run --dry-run` to confirm the execution slice without mutation.
- Use `plot --list` and `export --list` to confirm resolved outputs before
  generating them.

## 4. Execute only the slice you need

```bash
uv run reader run <config|dir|index>
uv run reader plot <config|dir|index>
uv run reader export <config|dir|index>
uv run reader notebook <config|dir|index> --mode none
uv run reader notebook <config|dir|index> --mode run --headless
```

`run` materializes records. `plot` and `export` materialize their output
surfaces after the experiment is ready. `notebook --mode none` scaffolds a
review notebook without launching Marimo, and `--mode run --headless` prints a
loopback URL for agent/browser review.

## 5. Verify outputs and provenance

```bash
uv run reader records <config|dir|index>
uv run reader inspect <config|dir|index>
```

Then inspect:

- `outputs/manifests/records.json`
- generated plot files
- generated export files

Verification should prove what ran and what was produced. It should not require
guessing from filesystem state alone.

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
