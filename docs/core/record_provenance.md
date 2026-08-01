---
doc_id: reader-record-provenance
surface: contract-reference
owner: reader-maintainers
last_verified: 2026-08-01
summary: Catalog-v4 epochs, schema-v6 record evidence, verification, and recovery.
---

# Record provenance and verification

Reader writes catalog schema v4 with schema-v6 dataframe and file-bundle
records. The catalog owns one canonical `provenance_epoch_id`; its active
invocation ledger is
`outputs/manifests/invocations/<provenance_epoch_id>.jsonl`. A current v6
record binds together:

- the normalized complete `reader/v8` config digest;
- the effective producer-config digest;
- Reader version and installed-source digest;
- each consumed file’s experiment-relative path, byte size, SHA-256, and
  selection policy;
- the exact upstream record revision consumed by downstream steps; and
- the exact experiment and record revision consumed through each cross-experiment
  record resource;
- each generated file’s output-relative path, byte size, and SHA-256.

Plot and export plugins still update their configured current files under
`outputs/plots/` or `outputs/exports/`. Each file-bundle record points to an
immutable revision under `outputs/artifacts/file_bundles/<phase>/<step>/`
(`__r2`, `__r3`, and so on for later revisions). A rerun can therefore advance
the current presentation files without invalidating any earlier catalog
history entry.

Run the verifier after producing or moving outputs:

```bash
uv run reader verify <config|dir|index>
uv run reader verify <config|dir|index> --format json
```

`records_ready` means every current record passed verification against the
current config, Reader build, sources, exact upstream revisions, and generated
files. `catalog_ready` means a valid catalog-schema-v4 envelope with
record-schema-v6 payloads exists but its recorded
config or Reader build differs from the current environment. A digest,
missing-file, exact-upstream-revision, or invalid-schema failure is `blocked`.

`reader records` is a catalog view. By default it projects the catalog through
the current configuration and, when outputs are declared, the current compiled
workbench. Records from a prior configuration or from renamed or retired steps
do not enter current handoffs. `reader records --all` retains access to those
catalog identities and their revision counts. Neither view replaces `reader
verify`.
Verification also audits the catalog-selected invocation-schema-v2 ledger.
Every invocation must contain one attempt and at most one terminal result;
every catalog revision must be claimed exactly once by a terminal result. An
attempt without a terminal result is reported as
`invocation.finalization_unconfirmed`; Reader keeps committed records intact
and reports the incomplete provenance rather than inventing a terminal state.

## Invalid catalog recovery

Reader reads and writes catalog schema v4 and record schema v6 only. A
catalog containing schema-v5 or older record payloads is invalid, not a
degraded compatibility mode.
Replace the generated catalog and perform a complete pipeline rerun from
staged source inputs:

```bash
uv run reader run <config|dir|index> --reset-records
```

`--reset-records` cannot be combined with `--dry-run`, `--from`, `--until`, or
`--only`. The reset stages Reader-owned generated state, initializes a
fresh epoch, and removes the prior dataframe artifacts, invocation ledgers,
execution log, plots, and exports as one bounded operation. Reader closes the
old log before staging it and opens a new log only after the epoch transaction
succeeds. If epoch initialization fails, Reader restores the staged state.
Generated notebooks and unrelated files at the output root are preserved. Plot
and export sinks must use dedicated subdirectories; ambiguous flattened sinks
fail before mutation. An existing `.reader-reset.*.staging` directory is
retained recovery evidence: inspect its `roots/`, `manifests/`, and `files/`
entries, restore the prior epoch or archive confirmed residue, then remove the
staging directory before running any mutating Reader operation. Reader does not
decode retired records or fabricate provenance by hashing older files; the
complete rerun writes current evidence.

Verification is scoped to records owned by the current compiled workbench.
Records from removed pipeline, plot, or export surfaces do not make the current
declaration unverifiable. Reader's fixed canonical notebook is a generated,
read-only viewport over verified records. The notebook file itself is operator
scaffolding, not a `RecordStore`-published file-bundle record.
