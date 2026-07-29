---
doc_id: reader-record-provenance
surface: contract-reference
owner: reader-maintainers
last_verified: 2026-07-28
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
- each generated file’s output-relative path, byte size, and SHA-256.

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

## Invalid catalog recovery

Reader reads and writes record schema v5 only. An older record payload is an
invalid catalog, not a degraded compatibility mode. Re-run the owning
experiment surface from staged source inputs to produce current evidence.
Reader does not fabricate provenance by hashing files left by an older run.
