---
doc_id: reader-dop-transfer-verification
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-07-28
summary: Safe input transfer, checksum, staging, and post-transfer verification procedure for Reader workspaces.
---

# Transfer and Verification

Use this page after the data class and metadata contract are clear.

Return to the [Data Operations Plan](../data_operations_plan.md) when you only
need the overview.

## Transfer Rules

- Put raw inputs in `inputs/` with the original filename when practical.
- Keep hand-authored notebooks in `notebooks/`; generated scaffolds belong in
  `outputs/notebooks/`.
- Keep generated records, plots, exports, and manifests under `outputs/`.
- Use explicit `resources` entries for files consumed by compiled steps.
  Protocol-owned discovery fields, such as cytometry `auto_roots`, own any
  directory scanning.
- When materializing from Google Drive or another external system, record the
  source and staged path in the handoff.
- Do not copy generated outputs from an old experiment into a new one. Copy
  config/metadata intent only when the new run is semantically close, then
  regenerate outputs.
- If source ownership, source location, or transfer status is unknown, leave the
  intake blocked instead of inventing a path that makes preflight look cleaner.

## Verification Commands

Record a SHA-256 digest before and after a transfer when the source can be read
as a local file:

```bash
shasum -a 256 <source-file>
shasum -a 256 <staged-file>
```

Matching values prove that the transfer preserved the file bytes. They do not
prove that the file was assigned to the right experiment or interpreted with
the right assay semantics.

Use the cheapest check that answers the next question, then broaden only when
the surface is ready:

```bash
uv run reader validate <config|dir|index> --no-files
uv run reader validate <config|dir|index>
uv run reader run <config|dir|index> --dry-run --format json
uv run reader plot <config|dir|index> --list
uv run reader export <config|dir|index> --list
uv run reader run <config|dir|index>
uv run reader records <config|dir|index>
uv run reader verify <config|dir|index>
```

## Evidence Bar

Verification is complete only when it proves:

- the config schema and protocol binding are valid;
- declared files/resources exist or the experiment is intentionally non-active;
- the compiled pipeline, plots, exports, and notebooks match the intended data
  class;
- `outputs/manifests/records.json` records the generated dataframe and
  file-bundle evidence;
- `reader verify` confirms the recorded files and their input evidence still
  match their recorded digests; and
- unresolved metadata assumptions are visible in the final handoff.
