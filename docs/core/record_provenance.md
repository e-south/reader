---
doc_id: reader-record-provenance
surface: contract-reference
owner: reader-maintainers
last_verified: 2026-07-29
summary: Schema-v5 record evidence, verification semantics, and invalid-catalog recovery.
---

# Record provenance and verification

Reader writes schema-v5 dataframe and file-bundle records. A current v5 record
binds together:

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
files. `catalog_ready` means a valid schema-v5 catalog exists but its recorded
config or Reader build differs from the current environment. A digest,
missing-file, exact-upstream-revision, or invalid-schema failure is `blocked`.

`reader records` is a catalog view. It does not replace `reader verify`.
Verification also audits `outputs/manifests/invocations.jsonl`: every invocation
must contain one attempt and at most one terminal result. An attempt without a
terminal result is reported as `invocation.finalization_unconfirmed`; Reader
keeps any committed records intact because their catalog and artifact evidence
remain the authoritative publication boundary.

## Invalid catalog recovery

Reader reads and writes record schema v5 only. An older record payload is an
invalid catalog, not a degraded compatibility mode. Replace the generated
catalog and perform a complete pipeline rerun from staged source inputs:

```bash
uv run reader run <config|dir|index> --reset-records
```

`--reset-records` cannot be combined with `--dry-run`, `--from`, `--until`, or
`--only`. It does not decode retired records or fabricate provenance by hashing
files left by an older run; the complete rerun writes current evidence.

Verification is scoped to records owned by the current compiled workbench.
Records from removed pipeline, plot, or export surfaces do not make the current
declaration unverifiable. Notebook bundles remain in scope while their template
is still declared.
