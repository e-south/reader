---
doc_id: reader-record-provenance
surface: contract-reference
owner: reader-maintainers
last_verified: 2026-07-29
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

`reader records` is a catalog view. It does not replace `reader verify`.
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
`--only`. The reset atomically replaces the catalog with an empty catalog and a
fresh epoch. The new active ledger begins separately; prior epoch ledgers are
retained as forensic residue and are not independently verifiable after their
catalog was replaced. Reader does not decode retired records or fabricate
provenance by hashing files left by an older run; the complete rerun writes
current evidence.

Verification is scoped to records owned by the current compiled workbench.
Records from removed pipeline, plot, or export surfaces do not make the current
declaration unverifiable. Reader's fixed canonical notebook is a generated,
read-only viewport over verified records. The notebook file itself is operator
scaffolding, not a `RecordStore`-published file-bundle record.
